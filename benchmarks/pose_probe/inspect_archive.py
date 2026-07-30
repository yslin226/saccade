"""Inspect a downloaded archive before extracting it.

A .tar.gz cannot run anything on its own; extraction is where the danger is.
An entry named ../../../.ssh/authorized_keys writes outside the target
directory, an absolute path ignores it entirely, a symlink can redirect later
writes at system files, and a 3GB archive can expand to fill a disk.

This lists the archive and reports on all of that without extracting a byte.
Run it, read the verdict, and only then extract.

Usage:
    uv run python -m benchmarks.pose_probe.inspect_archive path/to/file.tar.gz
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from pathlib import Path

__all__ = ["Report", "inspect"]

# Extensions that could execute if a user double-clicked them, or that a
# build step might pick up. A dataset of video frames has no business
# containing any of them.
EXECUTABLE_SUFFIXES = frozenset(
    {
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".bat",
        ".cmd",
        ".com",
        ".scr",
        ".ps1",
        ".sh",
        ".bash",
        ".zsh",
        ".vbs",
        ".js",
        ".jar",
        ".msi",
        ".app",
        ".deb",
        ".rpm",
        ".pkg",
        ".reg",
        ".lnk",
    }
)

# Formats that execute code when loaded by common Python libraries. .mat is
# not among them: scipy.io.loadmat parses the older MATLAB format as data,
# and v7.3 files are HDF5. Pickle is the one to fear.
DESERIALISATION_SUFFIXES = frozenset({".pkl", ".pickle", ".joblib", ".dill", ".pt", ".pth"})

# Above this ratio the archive expands more than a well-compressed dataset
# plausibly would, which is how a zip bomb looks from outside.
SUSPICIOUS_EXPANSION_RATIO = 200.0


class Report:
    """What the listing revealed."""

    def __init__(self) -> None:
        self.entries = 0
        self.total_unpacked = 0
        self.escaping: list[str] = []
        self.absolute: list[str] = []
        self.links: list[str] = []
        self.executables: list[str] = []
        self.deserialisation: list[str] = []
        self.suffixes: Counter[str] = Counter()
        self.top_level: set[str] = set()

    @property
    def safe(self) -> bool:
        return not (self.escaping or self.absolute or self.links or self.executables)

    def render(self, archive_size: int) -> str:
        lines = [
            f"entries          : {self.entries}",
            f"unpacked size    : {self.total_unpacked / 1e9:.2f} GB",
            f"archive size     : {archive_size / 1e9:.2f} GB",
        ]
        if archive_size:
            ratio = self.total_unpacked / archive_size
            flag = "  <-- SUSPICIOUS" if ratio > SUSPICIOUS_EXPANSION_RATIO else ""
            lines.append(f"expansion ratio  : {ratio:.1f}x{flag}")

        lines.append(f"top-level paths  : {sorted(self.top_level)[:6]}")
        lines.append("")
        lines.append("file types:")
        for suffix, count in self.suffixes.most_common(10):
            lines.append(f"  {suffix or '(none)':<12} {count:>8}")

        lines.append("")
        lines.append("=== safety checks ===")
        for label, found in (
            ("paths escaping the target directory", self.escaping),
            ("absolute paths", self.absolute),
            ("symlinks or hard links", self.links),
            ("executables", self.executables),
        ):
            if found:
                lines.append(f"  FAIL  {label}: {len(found)}")
                for name in found[:5]:
                    lines.append(f"          {name}")
            else:
                lines.append(f"  ok    no {label}")

        if self.deserialisation:
            lines.append(
                f"  note  {len(self.deserialisation)} file(s) that execute code when "
                f"loaded by some libraries — do not unpickle them:"
            )
            for name in self.deserialisation[:5]:
                lines.append(f"          {name}")

        lines.append("")
        lines.append("VERDICT: " + ("safe to extract" if self.safe else "DO NOT EXTRACT"))
        return "\n".join(lines)


def inspect(archive: Path) -> Report:
    """List an archive and report anything dangerous. Extracts nothing."""
    report = Report()

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            report.entries += 1
            report.total_unpacked += member.size

            name = member.name
            path = Path(name)

            if path.is_absolute() or name.startswith("/"):
                report.absolute.append(name)
            elif ".." in path.parts:
                report.escaping.append(name)

            if member.issym() or member.islnk():
                report.links.append(f"{name} -> {member.linkname}")

            suffix = path.suffix.lower()
            report.suffixes[suffix] += 1
            if suffix in EXECUTABLE_SUFFIXES:
                report.executables.append(name)
            if suffix in DESERIALISATION_SUFFIXES:
                report.deserialisation.append(name)

            if path.parts:
                report.top_level.add(path.parts[0])

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args(argv)

    if not args.archive.is_file():
        print(f"no such file: {args.archive}", file=sys.stderr)
        return 2

    print(f"inspecting {args.archive} (listing only, nothing extracted)")
    print()
    report = inspect(args.archive)
    print(report.render(args.archive.stat().st_size))
    return 0 if report.safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
