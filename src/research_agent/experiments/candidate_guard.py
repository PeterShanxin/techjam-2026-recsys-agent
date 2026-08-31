"""Write-confined launcher for untrusted candidate code.

Run as::

    python -B -s candidate_guard.py --write-root DIR [--read-root DIR]... \
        [--deny-read PATH]... --candidate FILE -- [candidate args]

The guard installs a CPython audit hook (:func:`sys.addaudithook`) and then
executes the candidate in this process. The hook is the real boundary: it sees
every filesystem and process primitive the interpreter performs, whatever the
source text looked like, so string obfuscation, ``getattr`` indirection,
``exec`` of a computed string, and dynamic imports are all equally covered.
CPython provides no way to remove an installed audit hook.

Five design points carry the security:

* **Events are default-deny.** Only events on ``ALLOWED_EVENTS`` /
  ``ALLOWED_PREFIXES`` -- none of which can touch the filesystem, spawn a
  process, or load native code -- pass. Enumerating *dangerous* events does
  not work: ``sqlite3.connect`` writes a database from C without raising a
  single ``open`` event, and on Windows ``shutil.copy2`` raises only
  ``_winapi.CopyFile2``. A real scorer raises 12 distinct events, so the
  allow list is small and stable.

* **Reads are allowlisted too, not just writes.** Otherwise stripping secrets
  from the environment is theatre: the candidate reads them straight back
  from ``/proc/<ppid>/environ``, ``~/.aws/credentials``, or the repo ``.env``.

* **Paths must be absolute.** POSIX ``*at`` syscalls resolve a relative path
  against a ``dir_fd`` while this hook would resolve it against the process
  cwd, so a relative path cannot be validated soundly. An absolute path makes
  ``dir_fd`` irrelevant by definition, which closes that class outright.

* **Nothing the hook calls is reachable from candidate code.** Every helper
  and every stdlib callable is captured as a closure local at install time.
  Module-level helpers would be rebindable through ``sys.modules``, and
  ``os.path.realpath`` would be rebindable through ``os`` -- either one lets a
  candidate make the hook validate a different path than the kernel acts on.

* **Anything unresolvable is denied**, never allowed through.

* **Subinterpreters are refused at import.** CPython raises no audit event
  when one is created, and hooks are per-interpreter and not copied, so code
  run there would have no boundary at all. The import event is the only
  enforcement point available for that capability.

* **The sandbox is writable or importable, never both.** Native code loaded
  out of a write root would run constructors that write with no Python event
  at all, so imports are refused when the module file, or any ``sys.path``
  entry, lives inside the writable tree.

Trusted dependencies are imported *before* the hook is installed: NumPy loads
``ctypes`` on Windows, so ``ctypes.*`` cannot be denied from process start.
Closing the gate by installing the hook, rather than by setting a flag, leaves
no enforcement switch in the process for candidate code to flip back.
"""
from __future__ import annotations

import os
import sys

# Events that cannot touch the filesystem, create a process, or load native
# code. Measured from real scorer runs, plus headroom for interpreter
# introspection and shutdown. Everything absent from here is denied.
ALLOWED_EVENTS = frozenset(
    {
        "compile",
        "exec",
        "import",
        "os.listdir",
        "os.scandir",
        "os.walk",
        "os.chdir",
        "os.getxattr",
        "os.listxattr",
        "os.putenv",
        "os.unsetenv",
        "glob.glob",
        "glob.glob/2",
        "mmap.__new__",  # a writable mapping still needs a permitted open()
        # Display-only hooks that fire while an exception is being reported.
        "sys.excepthook",
        "sys.unraisablehook",
        # The only sys.* event a real scorer raises. The rest of the namespace
        # is denied: sys._getframe, sys._current_frames, sys.settrace and
        # sys.setprofile all hand back frame objects, and on 3.13+ frame
        # f_locals writes through -- which reaches the hook's closure cells.
        "sys._getframemodulename",
        # Interpreter teardown, raised while the process is finalising. These
        # are the destruction half of the cpython.* namespace; creation
        # (PyInterpreterState_New) stays denied.
        "cpython.PyInterpreterState_Clear",
        "cpython.PyInterpreterState_Delete",
    }
)

# Namespaces that are pure computation. Deliberately NOT here: `cpython.`,
# because `cpython.PyInterpreterState_New` creates a subinterpreter and audit
# hooks are per-interpreter -- code run there would have no boundary at all.
# Also deliberately NOT here: `gc.`, whose
# get_objects/get_referrers enumerate live objects, which is enough to find
# the hook function and assign to `hook.__closure__[i].cell_contents`. Cells
# are writable, so that would let a candidate swap out the write roots, the
# read roots, or the allow list itself.
ALLOWED_PREFIXES = (
    "builtins.",
    "object.",
    "function.",
    "code.",
    "marshal.",
    "pickle.",
    "array.",
    "time.",
    "random.",
    "hashlib.",
    "decimal.",
)

# event -> indices of arguments naming a path this event *writes*. Source
# paths of copy operations are reads and are deliberately not listed;
# os.rename unlinks its source, so both sides count. Every other
# filesystem event is denied by the default-deny rule rather than listed.
WRITE_PATH_ARGS = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.utime": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
    "tempfile.mkstemp": (0,),
}


VIOLATION_EXIT_CODE = 3
VIOLATION_MARKER = "SandboxViolation"


# Modules denied by name. This is a name blocklist, which the rest of this
# module argues against -- it is used here because CPython raises *no* audit
# event for subinterpreter creation (verified on 3.14: `_interpreters.create()`
# emits nothing), and code inside a subinterpreter runs with no audit hook at
# all, since hooks are per-interpreter and are not copied. The import event is
# therefore the only enforcement point, and it is a sound one: every route to
# these modules -- direct, `importlib.import_module`, or transitively through
# `concurrent.futures` -- raises it.
DENIED_MODULES = frozenset(
    {
        "_interpreters",
        "_xxsubinterpreters",
        "_interpchannels",
        "_xxinterpchannels",
        "_interpqueues",
        "_xxinterpqueues",
        # _testcapi.run_in_subinterp is the same capability by another name,
        # and these ship as stdlib extensions under a permitted read root.
        "_testcapi",
        "_testinternalcapi",
        "_testmultiphase",
        "_testsinglephase",
        "_testlimitedcapi",
    }
)
DENIED_MODULE_PATHS = ("concurrent.interpreters",)


class SandboxViolation(PermissionError):
    """Candidate attempted an operation outside its sandbox."""


def install_guard(write_roots, read_roots=(), deny_read_paths=()) -> None:
    """Install the audit hook. Cannot be undone for the life of the process.

    Every name the hook touches is bound here as a closure local. Candidate
    code cannot reach these cells, so it cannot swap out the path-resolution
    the hook depends on.
    """
    # Captured now: `os.path.realpath` and friends are rebindable attributes,
    # and module-level helpers are reachable through sys.modules.
    _realpath = os.path.realpath
    _normcase = os.path.normcase
    _isabs = os.path.isabs
    _isinstance = isinstance
    _type = type
    _sep = os.sep
    _fsencoding = sys.getfilesystemencoding()
    _write_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_EXCL", 0)
    )
    _write_mode_chars = frozenset("wax+")
    _allowed_events = ALLOWED_EVENTS
    _allowed_prefixes = ALLOWED_PREFIXES
    _write_path_args = WRITE_PATH_ARGS
    _exit = os._exit
    _denied_modules = DENIED_MODULES
    _denied_module_paths = DENIED_MODULE_PATHS
    # Bound now, not looked up per call: `sys.stderr.write` is assignable, so
    # a late-bound lookup would run candidate code with the hook frame live on
    # the stack -- and from there `f_back` reaches the cells below.
    _stderr_write = sys.stderr.write
    _stderr_flush = sys.stderr.flush
    _linesep = os.linesep
    _prefix = VIOLATION_MARKER + ": "
    _exit_code = VIOLATION_EXIT_CODE

    def deny(message):
        """End the process rather than raising into candidate code.

        A raised exception carries `__traceback__.tb_frame`, and on 3.13+ a
        frame's f_locals writes through to the closure cells behind it -- so
        catching one violation would hand the candidate the write roots, the
        read roots, and the allow list. Exiting removes that surface, and a
        candidate that reaches for an escape has no legitimate recovery path
        anyway. The runner reports the attempt as failed.

        Nothing here may run candidate code: every name is captured, and
        `message` is always a str this module built. No str()/repr() is
        applied to a candidate-controlled value anywhere on this path.
        """
        try:
            _stderr_write(_prefix + message + _linesep)
            _stderr_flush()
        except BaseException:  # noqa: BLE001 - never let reporting block the exit
            pass
        _exit(_exit_code)

    def as_text(value):
        """A plain str safe to put in a message.

        Exact type check, not isinstance: a str *subclass* defining __radd__
        would have its method called before str.__add__ when concatenated onto
        a literal, running candidate code with the hook frame on the stack --
        the same re-entry class as __repr__ and __fspath__.
        """
        return value if _type(value) is str else "<non-text value>"

    def canonical(value):
        """Absolute, symlink-resolved, case-folded path, or None.

        Only str and bytes are accepted. Calling os.fspath on an arbitrary
        object would run its __fspath__ inside this hook -- `open` and `os.*`
        hand the audit event an already-normalised str, but `shutil.*` events
        pass whatever the caller supplied.
        """
        raw = value
        if _isinstance(raw, bytes):
            try:
                raw = raw.decode(_fsencoding, "surrogateescape")
            except (UnicodeDecodeError, LookupError):
                return None
        if not _isinstance(raw, str) or not raw:
            return None
        # Relative paths cannot be validated: a POSIX *at syscall would
        # resolve them against a dir_fd instead of the cwd used here.
        if not _isabs(raw):
            return None
        try:
            return _normcase(_realpath(raw))
        except (OSError, ValueError):
            return None

    def normalise(roots):
        out = set()
        for root in roots:
            try:
                out.add(_normcase(_realpath(str(root))))
            except (OSError, ValueError):
                continue
        return tuple(sorted(out))

    writable = normalise(write_roots)
    if not writable:
        raise ValueError("candidate guard requires at least one writable root")
    # sys.path and the interpreter prefixes cover the stdlib, site-packages,
    # and every PYTHONPATH entry the candidate legitimately imports from.
    readable = normalise(
        [
            *write_roots,
            *read_roots,
            *[entry for entry in sys.path if entry],
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            sys.base_exec_prefix,
        ]
    )
    denied_reads = normalise(deny_read_paths)

    def contained(path, roots):
        for root in roots:
            if path == root or path.startswith(root + _sep):
                return True
        return False

    def hook(event, args):
        if event == "open":
            target = args[0] if args else None
            if _isinstance(target, int):
                return  # existing descriptor; its open() was already checked
            path = canonical(target)
            if path is None:
                deny("candidate may not open an unresolvable path: " + as_text(target))
            if contained(path, denied_reads):
                deny("candidate may not read " + as_text(target))
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            writing = (_isinstance(flags, int) and flags & _write_flags) or (
                _isinstance(mode, str) and _write_mode_chars.intersection(mode)
            )
            if writing:
                if not contained(path, writable):
                    deny("candidate may not write outside its sandbox: " + as_text(target))
            elif not contained(path, readable):
                deny("candidate may not read outside its sandbox: " + as_text(target))
            return

        indices = _write_path_args.get(event)
        if indices is not None:
            for index in indices:
                if index >= len(args):
                    continue
                value = args[index]
                if value is None:
                    continue
                path = canonical(value)
                if path is None:
                    deny("candidate may not " + event + " on " + as_text(value))
                if not contained(path, writable):
                    deny("candidate may not " + event + " outside its sandbox: "
                         + as_text(value))
            return

        if event == "import":
            module = args[0] if args else None
            if _isinstance(module, str) and module:
                if module.split(".", 1)[0] in _denied_modules or module.startswith(
                    _denied_module_paths
                ):
                    deny("candidate may not import " + as_text(module)
                         + " (subinterpreters escape the hook)")
            # A candidate that writes a shared library into its sandbox and
            # imports it runs native constructors, which fopen/write with no
            # Python event at all. Both routes are visible here: a direct
            # ExtensionFileLoader populates `filename`, while a sys.path
            # import leaves it None but exposes the search path.
            filename = args[1] if len(args) > 1 else None
            if filename is not None:
                loaded = canonical(filename)
                # Fail closed: a relative origin resolves against cwd, which
                # the runner sets to a write root, so an unresolvable module
                # path is exactly the escape this branch exists to stop.
                if loaded is None:
                    deny("candidate may not import from a relative path: " + as_text(filename))
                if contained(loaded, writable):
                    deny("candidate may not import from its writable sandbox: " + as_text(filename))
            for entry in (args[2] if len(args) > 2 else None) or ():
                resolved = canonical(entry) if entry else None
                if resolved is None:
                    deny("candidate may not import with a relative sys.path entry: " + as_text(entry))
                if contained(resolved, writable):
                    deny("candidate may not put a writable directory on sys.path: " + as_text(entry))
            return

        if event == "os.add_dll_directory":
            # Legitimate NumPy use happens before the hook exists; afterwards
            # this would only ever widen the native search path.
            added = canonical(args[0]) if args else None
            if added is None or contained(added, writable):
                deny("candidate may not add a sandbox directory to the DLL path")
            return

        if event in _allowed_events or event.startswith(_allowed_prefixes):
            return
        deny("candidate may not use " + event + " (not permitted in sandbox)")

    sys.addaudithook(hook)


def _preimport(modules) -> None:
    """Load trusted dependencies before the gate closes.

    NumPy imports ctypes on Windows; the starter modules are the candidate's
    supported data/evaluation interface. Failures are silent because a missing
    optional module must surface as the candidate's own ImportError later.
    """
    import importlib

    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - candidate reports its own import errors
            pass


def _parse_argv(argv):
    write_roots, read_roots, deny_reads, preimport = [], [], [], []
    candidate = None
    rest = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            rest = list(argv[index + 1:])
            break
        if index + 1 >= len(argv):
            raise SystemExit(f"candidate_guard: missing value for {token}")
        value = argv[index + 1]
        if token == "--write-root":
            write_roots.append(value)
        elif token == "--read-root":
            read_roots.append(value)
        elif token == "--deny-read":
            deny_reads.append(value)
        elif token == "--preimport":
            preimport.append(value)
        elif token == "--candidate":
            candidate = value
        else:
            raise SystemExit(f"candidate_guard: unknown option {token}")
        index += 2
    if candidate is None:
        raise SystemExit("candidate_guard: --candidate is required")
    return candidate, write_roots, read_roots, deny_reads, preimport, rest


def main(argv=None) -> int:
    candidate, write_roots, read_roots, deny_reads, preimport, rest = _parse_argv(
        list(sys.argv[1:] if argv is None else argv)
    )
    sys.dont_write_bytecode = True
    _preimport(preimport)

    import runpy

    candidate = os.path.abspath(candidate)
    sys.argv = [candidate, *rest]
    # Drop the guard's own directory so it cannot shadow candidate imports.
    if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
        sys.path.pop(0)

    # The candidate file itself, not its directory: sibling files in a shared
    # workspace are none of its business.
    install_guard(write_roots, [*read_roots, candidate], deny_reads)
    runpy.run_path(candidate, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
