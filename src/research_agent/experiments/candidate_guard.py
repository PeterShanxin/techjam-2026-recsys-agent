"""Write-confined launcher for untrusted candidate code.

Run as::

    python -B -s candidate_guard.py --write-root DIR [--deny-read PATH]... \
        --candidate FILE -- [candidate args]

The guard installs a CPython audit hook (:func:`sys.addaudithook`) and then
executes the candidate in this process. The hook is the real boundary: it sees
every filesystem and process primitive the interpreter performs, whatever the
source text looked like, so string obfuscation, ``getattr`` indirection,
``exec`` of a computed string, and dynamic imports are all equally covered.
CPython provides no way to remove an installed audit hook.

Two design points matter for correctness:

* Trusted dependencies (NumPy, the starter modules) are imported *before* the
  hook is installed. NumPy loads ``ctypes`` on Windows, so ``ctypes.*`` cannot
  be denied from process start. Because the gate closes by installing the
  hook rather than by setting a flag, there is no enforcement switch left in
  the process for candidate code to reach and flip back.
* Paths are compared after ``os.path.realpath``, so ``..`` traversal, symlinks,
  and Windows junctions resolve to their true target before the containment
  check runs.

Unknown audit events inside filesystem/process namespaces are denied rather
than allowed, so a primitive this module has not enumerated fails closed.
``shutil.copy2`` on Windows is the worked example: it raises only
``_winapi.CopyFile2`` and no ``open`` event at all.
"""
from __future__ import annotations

import os
import sys

WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC | getattr(os, "O_EXCL", 0)
)
_WRITE_MODE_CHARS = frozenset("wax+")

# Capability classes an experiment candidate never needs. Denying process
# creation keeps the candidate inside this guarded interpreter; denying
# ctypes stops raw syscalls that would bypass the audit hook; denying network
# egress means anything the candidate does learn cannot leave the machine.
DENIED_PREFIXES = (
    "ctypes.",
    "subprocess.",
    "socket.",
    "urllib.",
    "ftplib.",
    "smtplib.",
    "imaplib.",
    "poplib.",
    "telnetlib.",
    "http.client.",
    "webbrowser.",
    "winreg.",
    "pdb.",
    "_posixsubprocess.",
)
DENIED_EVENTS = frozenset(
    {
        "os.system",
        "os.exec",
        "os.spawn",
        "os.posix_spawn",
        "os.fork",
        "os.forkpty",
        "os.startfile",
        "os.startfile/2",
        # Links are the classic escape: create one inside the sandbox, write
        # through it to a target outside. Candidates have no use for them.
        "os.symlink",
        "os.link",
        "_winapi.CreateJunction",
    }
)

# Namespaces where an event this module does not recognise is denied.
GUARDED_PREFIXES = (
    "os.",
    "shutil.",
    "_winapi.",
    "msvcrt.",
    "nt.",
    "fcntl.",
    "pty.",
    "tempfile.",
)

# Events inside guarded namespaces that cannot write: allowed explicitly.
NEUTRAL_EVENTS = frozenset(
    {
        "os.listdir",
        "os.scandir",
        "os.walk",
        "os.chdir",
        "os.getxattr",
        "os.listxattr",
        "os.add_dll_directory",  # NumPy uses this to find its native libraries
        "os.putenv",
        "os.unsetenv",
        "mmap.__new__",  # a writable mapping still needs a permitted open()
    }
)

# event -> indices of arguments naming a path this event *writes*. Source
# paths of copy operations are reads and are deliberately not listed;
# os.rename unlinks its source, so both sides count.
WRITE_PATH_ARGS = {
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    "os.rename": (0, 1),
    "os.truncate": (0,),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.utime": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
    "shutil.rmtree": (0,),
    "shutil.unpack_archive": (1,),
    "tempfile.mkstemp": (0,),
    "_winapi.CopyFile2": (1,),
    "_winapi.CreateFile": (0,),
    "_winapi.CreateNamedPipe": (0,),
}


class SandboxViolation(PermissionError):
    """Candidate attempted an operation outside its sandbox."""


def _normalise(path: object) -> str | None:
    """Absolute, symlink-resolved, case-folded path, or None if not a path."""
    if isinstance(path, int):
        return None  # already-open descriptor; its open() was checked
    try:
        raw = os.fspath(path)  # type: ignore[arg-type]
    except TypeError:
        return None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode(sys.getfilesystemencoding(), "surrogateescape")
        except (UnicodeDecodeError, LookupError):
            return None
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return os.path.normcase(os.path.realpath(raw))
    except (OSError, ValueError):
        return None


def _contained(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + os.sep) for root in roots)


def _wants_write(mode: object, flags: object) -> bool:
    if isinstance(flags, int) and flags & WRITE_FLAGS:
        return True
    if isinstance(mode, str) and _WRITE_MODE_CHARS.intersection(mode):
        return True
    return False


def install_guard(write_roots, deny_read_paths=()) -> None:
    """Install the audit hook. Cannot be undone for the life of the process."""
    roots = tuple(sorted({os.path.normcase(os.path.realpath(str(r))) for r in write_roots}))
    if not roots:
        raise ValueError("candidate guard requires at least one writable root")
    denied_reads = tuple(
        sorted({os.path.normcase(os.path.realpath(str(p))) for p in deny_read_paths})
    )

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = args[0] if args else None
            resolved = _normalise(path)
            if resolved is None:
                return
            if denied_reads and _contained(resolved, denied_reads):
                raise SandboxViolation(f"candidate may not read {path!r}")
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else None
            if _wants_write(mode, flags) and not _contained(resolved, roots):
                raise SandboxViolation(
                    f"candidate may not write outside its sandbox: {path!r}"
                )
            return

        indices = WRITE_PATH_ARGS.get(event)
        if indices is not None:
            for index in indices:
                if index >= len(args):
                    continue
                resolved = _normalise(args[index])
                if resolved is None:
                    continue
                if not _contained(resolved, roots):
                    raise SandboxViolation(
                        f"candidate may not {event} outside its sandbox: {args[index]!r}"
                    )
            return

        if event in NEUTRAL_EVENTS:
            return
        if event in DENIED_EVENTS or event.startswith(DENIED_PREFIXES):
            raise SandboxViolation(f"candidate may not use {event}")
        if event.startswith(GUARDED_PREFIXES):
            # Unrecognised filesystem/process primitive: fail closed.
            raise SandboxViolation(f"candidate may not use {event} (not permitted in sandbox)")

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
    write_roots, deny_reads, preimport = [], [], []
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
    return candidate, write_roots, deny_reads, preimport, rest


def main(argv=None) -> int:
    candidate, write_roots, deny_reads, preimport, rest = _parse_argv(
        list(sys.argv[1:] if argv is None else argv)
    )
    sys.dont_write_bytecode = True
    _preimport(preimport)

    install_guard(write_roots, deny_reads)

    import runpy

    sys.argv = [candidate, *rest]
    # Drop the guard's own directory so it cannot shadow candidate imports.
    if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
        sys.path.pop(0)
    runpy.run_path(candidate, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
