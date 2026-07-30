"""Tests for the archive inspector.

Its whole job is to refuse dangerous archives, so the tests build dangerous
archives and check it says no. A scanner that has never rejected anything is
not a scanner.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

from benchmarks.pose_probe.inspect_archive import inspect


def build(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    """Write a .tar.gz containing the given files."""
    with tarfile.open(path, "w:gz") as tar:
        for name, data in entries:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def build_with_member(path: Path, info: tarfile.TarInfo) -> Path:
    """Write an archive containing one crafted member."""
    with tarfile.open(path, "w:gz") as tar:
        tar.addfile(info)
    return path


class TestHarmlessArchives:
    def test_a_dataset_shaped_archive_passes(self, tmp_path: Path) -> None:
        archive = build(
            tmp_path / "clean.tar.gz",
            [
                ("Penn_Action/frames/0001/000001.jpg", b"\xff\xd8\xff" + b"x" * 500),
                ("Penn_Action/frames/0001/000002.jpg", b"\xff\xd8\xff" + b"x" * 500),
                ("Penn_Action/labels/0001.mat", b"MATLAB 5.0" + b"y" * 200),
            ],
        )
        report = inspect(archive)

        assert report.safe is True
        assert report.entries == 3
        assert report.suffixes[".jpg"] == 2
        assert report.top_level == {"Penn_Action"}

    def test_mat_files_are_not_flagged_as_dangerous(self, tmp_path: Path) -> None:
        """scipy.io.loadmat parses MATLAB files as data, unlike pickle."""
        archive = build(tmp_path / "mats.tar.gz", [("labels/0001.mat", b"MATLAB")])
        report = inspect(archive)

        assert report.safe is True
        assert report.deserialisation == []


class TestPathTraversal:
    def test_a_parent_directory_escape_is_rejected(self, tmp_path: Path) -> None:
        archive = build(
            tmp_path / "escape.tar.gz",
            [("../../../.ssh/authorized_keys", b"ssh-rsa AAAA")],
        )
        report = inspect(archive)

        assert report.safe is False
        assert len(report.escaping) == 1

    def test_an_escape_buried_mid_path_is_rejected(self, tmp_path: Path) -> None:
        archive = build(
            tmp_path / "buried.tar.gz",
            [("Penn_Action/frames/../../../etc/cron.d/evil", b"* * * * * root sh")],
        )
        report = inspect(archive)
        assert report.safe is False
        assert report.escaping

    def test_an_absolute_path_is_rejected(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "absolute.tar.gz", [("/etc/passwd", b"root:x:0:0")])
        report = inspect(archive)

        assert report.safe is False
        assert len(report.absolute) == 1


class TestLinks:
    def test_a_symlink_is_rejected(self, tmp_path: Path) -> None:
        info = tarfile.TarInfo(name="Penn_Action/shortcut")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/shadow"

        report = inspect(build_with_member(tmp_path / "sym.tar.gz", info))

        assert report.safe is False
        assert "/etc/shadow" in report.links[0]

    def test_a_hard_link_is_rejected(self, tmp_path: Path) -> None:
        info = tarfile.TarInfo(name="Penn_Action/hard")
        info.type = tarfile.LNKTYPE
        info.linkname = "../../../.bashrc"

        report = inspect(build_with_member(tmp_path / "hard.tar.gz", info))
        assert report.safe is False
        assert report.links


class TestExecutables:
    def test_an_executable_is_rejected(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "exe.tar.gz", [("Penn_Action/setup.exe", b"MZ")])
        report = inspect(archive)

        assert report.safe is False
        assert report.executables == ["Penn_Action/setup.exe"]

    def test_a_shell_script_is_rejected(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "sh.tar.gz", [("tools/run.sh", b"#!/bin/sh\nrm -rf /")])
        report = inspect(archive)
        assert report.safe is False

    def test_a_dll_is_rejected(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "dll.tar.gz", [("lib/helper.dll", b"MZ")])
        assert inspect(archive).safe is False


class TestPickles:
    def test_a_pickle_is_noted_without_failing_the_archive(self, tmp_path: Path) -> None:
        """Loading it is the hazard, extracting it is not."""
        archive = build(tmp_path / "pkl.tar.gz", [("data/cache.pkl", b"\x80\x04")])
        report = inspect(archive)

        assert report.deserialisation == ["data/cache.pkl"]
        assert report.safe is True


class TestExpansion:
    def test_the_unpacked_size_is_measured(self, tmp_path: Path) -> None:
        """So a zip bomb can be spotted before it fills the disk."""
        archive = build(tmp_path / "big.tar.gz", [("data/zeros.bin", b"\x00" * 100_000)])
        report = inspect(archive)

        assert report.total_unpacked == 100_000
        # Highly compressible content, so the ratio should be large.
        assert report.total_unpacked > archive.stat().st_size

    def test_the_report_flags_an_extreme_ratio(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "bomb.tar.gz", [("z", b"\x00" * 5_000_000)])
        rendered = inspect(archive).render(archive.stat().st_size)
        assert "SUSPICIOUS" in rendered


class TestVerdict:
    def test_a_clean_archive_reads_as_safe(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "ok.tar.gz", [("frames/a.jpg", b"\xff\xd8")])
        assert "safe to extract" in inspect(archive).render(archive.stat().st_size)

    def test_a_dirty_archive_says_do_not_extract(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "bad.tar.gz", [("../evil", b"x")])
        assert "DO NOT EXTRACT" in inspect(archive).render(archive.stat().st_size)
