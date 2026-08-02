"""Task 1: the package exists, imports, and its layers point the right way.

The dependency-direction test is the one worth having. Clean Architecture is
a claim about which module may name which, and nothing else in the build
checks it — the engine's own guard scans ``src/saccade`` and deliberately
ignores this package, because rule 2 exists to keep MediaPipe out of the
engine, not out of the application that supplies it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import sandlot

SRC = Path(sandlot.__file__).resolve().parent

# Which layers each may import from. Inward only: a layer may name the ones
# beneath it and never the ones above.
ALLOWED: dict[str, set[str]] = {
    "domain": set(),
    "application": {"domain"},
    "infrastructure": {"domain", "application"},
    "interfaces": {"domain", "application", "infrastructure"},
}

LAYERS = frozenset(ALLOWED)

# Detectors and image libraries. The engine may not import these at all;
# this application may, but only in the layer whose job is the outside world.
VISION_PACKAGES = frozenset({"mediapipe", "ultralytics", "cv2"})


def modules_in(layer: str) -> list[Path]:
    return sorted((SRC / layer).rglob("*.py"))


def imported_layers(path: Path) -> set[str]:
    """Which sibling layers this module imports from."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
        elif isinstance(node, ast.Import):
            parts = node.names[0].name.split(".")
        else:
            continue

        if parts[0] == "sandlot" and len(parts) > 1 and parts[1] in LAYERS:
            found.add(parts[1])

    return found


def imported_roots(path: Path) -> set[str]:
    """Top-level package names this module imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)

    return roots


class TestThePackageExists:
    def test_it_imports(self) -> None:
        assert sandlot.__version__

    @pytest.mark.parametrize("layer", sorted(LAYERS))
    def test_every_layer_is_a_package(self, layer: str) -> None:
        assert (SRC / layer / "__init__.py").is_file()

    def test_it_ships_a_typing_marker(self) -> None:
        assert (SRC / "py.typed").is_file()

    def test_the_engine_is_importable_from_here(self) -> None:
        """The workspace source, not a release. If this fails, uv resolved
        saccade-vision from PyPI — where it is not published."""
        import saccade

        assert saccade.ActiveVisionAgent is not None


class TestDependenciesPointInward:
    """domain knows nothing; interfaces may know everything below it."""

    @pytest.mark.parametrize("layer", sorted(LAYERS))
    def test_a_layer_imports_only_from_below(self, layer: str) -> None:
        for path in modules_in(layer):
            forbidden = imported_layers(path) - ALLOWED[layer] - {layer}
            assert not forbidden, f"{path.relative_to(SRC)} imports {sorted(forbidden)}"

    def test_domain_imports_no_other_layer(self) -> None:
        """Stated separately because it is the load-bearing one: a domain
        module that reaches for a use case has put a rule where the domain
        tests cannot see it.

        Importing within ``domain`` is not that — a package re-exporting its
        own modules is how ``__init__`` works.
        """
        for path in modules_in("domain"):
            assert imported_layers(path) <= {"domain"}, path.relative_to(SRC)

    def test_the_scan_would_catch_a_violation(self, tmp_path: Path) -> None:
        """A guard nobody has seen fail is not a guard."""
        offender = tmp_path / "leak.py"
        offender.write_text("from sandlot.infrastructure.vision import x\n", encoding="utf-8")
        assert imported_layers(offender) == {"infrastructure"}


class TestDetectorsStayInInfrastructure:
    """Rule 2 keeps these out of the engine. Here they are allowed, but only
    where the outside world belongs — a domain module importing cv2 would
    make a metric untestable without a decoder."""

    @pytest.mark.parametrize("layer", ["domain", "application", "interfaces"])
    def test_no_vision_package_outside_infrastructure(self, layer: str) -> None:
        for path in modules_in(layer):
            found = imported_roots(path) & VISION_PACKAGES
            assert not found, f"{path.relative_to(SRC)} imports {sorted(found)}"

    def test_the_scan_would_catch_a_violation(self, tmp_path: Path) -> None:
        offender = tmp_path / "leak.py"
        offender.write_text("import mediapipe\n", encoding="utf-8")
        assert imported_roots(offender) & VISION_PACKAGES == {"mediapipe"}


class TestTheCLI:
    def test_it_reports_its_version(self) -> None:
        from sandlot.interfaces.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(["--version"])
        assert exit_info.value.code == 0

    def test_no_arguments_prints_help_and_fails(self) -> None:
        from sandlot.interfaces.cli import main

        assert main([]) == 1

    def test_a_missing_video_fails_rather_than_reporting_nothing(self, tmp_path: Path) -> None:
        """A command that appears to succeed while measuring nothing is worse
        than one that refuses. Task 1 asserted a placeholder here; the
        behaviour is now real and the exit code is UNREADABLE."""
        from sandlot.interfaces.cli import UNREADABLE, main

        assert main(["--data-dir", str(tmp_path), "analyze", "absent.mov"]) == UNREADABLE

    def test_the_data_dir_defaults_outside_the_repo(self) -> None:
        """So analysing a video never leaves files in a checkout."""
        from sandlot.interfaces.cli import DEFAULT_DATA_DIR

        assert DEFAULT_DATA_DIR.is_absolute()
        assert Path.cwd() not in DEFAULT_DATA_DIR.parents

    def test_the_data_dir_is_overridable(self, tmp_path: Path) -> None:
        from sandlot.interfaces.cli import build_parser

        args = build_parser().parse_args(["--data-dir", str(tmp_path), "analyze", "v.mov"])
        assert args.data_dir == tmp_path
