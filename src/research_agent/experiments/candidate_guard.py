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
    }
)

# Namespaces that are pure computation. Deliberately NOT here: `gc.`, whose
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
    "cpython.",
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
    _fspath = os.fspath
    _isinstance = isinstance
    _sep = os.sep
    _fsencoding = sys.getfilesystemencoding()
    _write_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_EXCL", 0)
    )
    _write_mode_chars = frozenset("wax+")
    _allowed_events = ALLOWED_EVENTS
    _allowed_prefixes = ALLOWED_PREFIXES
    _write_path_args = WRITE_PATH_ARGS
    _violation = SandboxViolation

    def canonical(value):
        """Absolute, symlink-resolved, case-folded path, or None."""
        try:
            raw = _fspath(value)
        except TypeError:
            return None
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
                raise _violation(f"candidate may not open an unresolvable path: {target!r}")
            if contained(path, denied_reads):
                raise _violation(f"candidate may not read {target!r}")
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            writing = (_isinstance(flags, int) and flags & _write_flags) or (
                _isinstance(mode, str) and _write_mode_chars.intersection(mode)
            )
            if writing:
                if not contained(path, writable):
                    raise _violation(
                        f"candidate may not write outside its sandbox: {target!r}"
                    )
            elif not contained(path, readable):
                raise _violation(f"candidate may not read outside its sandbox: {target!r}")
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
                    raise _violation(f"candidate may not {event} on {value!r}")
                if not contained(path, writable):
                    raise _violation(
                        f"candidate may not {event} outside its sandbox: {value!r}"
                    )
            return

        if event == "import":
            # A candidate that writes a shared library into its sandbox and
            # imports it runs native constructors, which fopen/write with no
            # Python event at all. Both routes are visible here: a direct
            # ExtensionFileLoader populates `filename`, while a sys.path
            # import leaves it None but exposes the search path.
            filename = args[1] if len(args) > 1 else None
            if filename is not None:
                loaded = canonical(filename)
                if loaded is not None and contained(loaded, writable):
                    raise _violation(
                        f"candidate may not import from its writable sandbox: {filename!r}"
                    )
            for entry in (args[2] if len(args) > 2 else None) or ():
                if not entry or entry == ".":
                    raise _violation("candidate may not import with cwd on sys.path")
                resolved = canonical(entry)
                if resolved is not None and contained(resolved, writable):
                    raise _violation(
                        f"candidate may not put a writable directory on sys.path: {entry!r}"
                    )
            return

        if event == "os.add_dll_directory":
            # Legitimate NumPy use happens before the hook exists; afterwards
            # this would only ever widen the native search path.
            added = canonical(args[0]) if args else None
            if added is None or contained(added, writable):
                raise _violation("candidate may not add a sandbox directory to the DLL path")
            return

        if event in _allowed_events or event.startswith(_allowed_prefixes):
            return
        raise _violation(f"candidate may not use {event} (not permitted in sandbox)")

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
