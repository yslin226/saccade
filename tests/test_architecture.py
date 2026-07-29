"""Architecture guard — the automated half of CLAUDE.md.

Rules 2 and 3 are the ones that rot quietly: one convenient ``import
mediapipe`` inside the engine, or one ``Image.open`` inside a pure module,
and the library stops being general without anyone noticing. Reviews miss
that. An AST scan does not.

These checks read source, they do not import it, so a violation is caught
even in a module that would fail to import.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "saccade"

# Rule 2: domain capability enters through register_tool(), never through an import.
FORBIDDEN_IMPORTS = {
    "mediapipe",
    "ultralytics",
    "sklearn",
    "scikit_learn",
    "torch",
    "tensorflow",
    "keras",
    "yolo",
}

# Rule 3: these modules reason about data handed to them; they never fetch it.
# The visual actions belong here too: they transform images passed in, and a
# crop that could read from disk would put file paths into the evidence chain.
PURE_MODULES = ("_planner.py", "_verifier.py", "_evidence.py", "geometry/", "actions/")

FORBIDDEN_CALLS = {
    "cv2.imread",
    "cv2.imwrite",
    "cv2.VideoCapture",
    "cv2.VideoWriter",
    "Image.open",
    "Image.save",
    "open",
}


def python_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def relative(path: Path) -> str:
    return path.relative_to(SRC).as_posix()


def is_pure_module(path: Path) -> bool:
    rel = relative(path)
    return any(rel.endswith(marker) or rel.startswith(marker) for marker in PURE_MODULES)


def imported_root_modules(tree: ast.Module) -> set[str]:
    """Every top-level package name imported by a module."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def called_names(tree: ast.Module) -> list[tuple[str, int]]:
    """Every call in a module, rendered as a dotted name with its line number."""
    calls: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            calls.append((func.id, node.lineno))
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            calls.append((f"{func.value.id}.{func.attr}", node.lineno))
        elif isinstance(func, ast.Attribute):
            calls.append((func.attr, node.lineno))
    return calls


class TestRule2NoDomainImports:
    """Saccade must not import MediaPipe, YOLO or any domain package."""

    @pytest.mark.parametrize("path", python_files(), ids=relative)
    def test_module_imports_no_domain_package(self, path: Path) -> None:
        found = imported_root_modules(parse(path)) & FORBIDDEN_IMPORTS
        assert not found, (
            f"{relative(path)} imports {sorted(found)}. Domain capability must be "
            f"injected via register_tool() — see CLAUDE.md rule 2."
        )

    def test_the_scan_actually_finds_a_violation(self, tmp_path: Path) -> None:
        """A guard that cannot fail is not a guard."""
        offender = tmp_path / "bad.py"
        offender.write_text("import mediapipe\n", encoding="utf-8")
        assert imported_root_modules(parse(offender)) & FORBIDDEN_IMPORTS == {"mediapipe"}

    def test_the_scan_catches_from_imports_too(self, tmp_path: Path) -> None:
        offender = tmp_path / "bad.py"
        offender.write_text("from ultralytics import YOLO\n", encoding="utf-8")
        assert imported_root_modules(parse(offender)) & FORBIDDEN_IMPORTS == {"ultralytics"}


class TestRule3NoIOInPureModules:
    """Pure logic modules receive data; they never load it."""

    @pytest.mark.parametrize(
        "path",
        [p for p in python_files() if is_pure_module(p)],
        ids=relative,
    )
    def test_pure_module_performs_no_file_io(self, path: Path) -> None:
        violations = [
            f"{name} (line {line})"
            for name, line in called_names(parse(path))
            if name in FORBIDDEN_CALLS
        ]
        assert not violations, (
            f"{relative(path)} performs file I/O: {violations}. Pure modules take "
            f"in-memory objects only — see CLAUDE.md rule 3."
        )

    def test_at_least_one_pure_module_is_being_scanned(self) -> None:
        """Guards against the parametrisation silently matching nothing."""
        scanned = [relative(p) for p in python_files() if is_pure_module(p)]
        assert scanned, "no pure modules matched PURE_MODULES — the guard is inert"
        assert any(p.startswith("geometry/") for p in scanned)

    def test_the_scan_actually_finds_a_violation(self, tmp_path: Path) -> None:
        offender = tmp_path / "shapes.py"
        offender.write_text("import cv2\nimg = cv2.imread('x.png')\n", encoding="utf-8")
        found = [name for name, _ in called_names(parse(offender)) if name in FORBIDDEN_CALLS]
        assert found == ["cv2.imread"]

    def test_the_scan_catches_bare_open(self, tmp_path: Path) -> None:
        offender = tmp_path / "shapes.py"
        offender.write_text("data = open('x.txt').read()\n", encoding="utf-8")
        found = [name for name, _ in called_names(parse(offender)) if name in FORBIDDEN_CALLS]
        assert found == ["open"]

    def test_cache_module_is_allowed_to_do_io(self) -> None:
        """The cache is not a pure module: persistence is its whole job."""
        assert not is_pure_module(SRC / "vlm" / "_cache.py")


class TestRule4PublicApiIsReal:
    """Every name in __all__ must exist and be importable."""

    def test_all_names_resolve(self) -> None:
        import saccade

        missing = [name for name in saccade.__all__ if not hasattr(saccade, name)]
        assert not missing, f"__all__ names missing from the package: {missing}"

    def test_all_is_sorted_and_unique(self) -> None:
        import saccade

        assert len(saccade.__all__) == len(set(saccade.__all__)), "duplicate names in __all__"
        assert saccade.__all__ == sorted(saccade.__all__), "__all__ should stay sorted"

    def test_every_public_name_is_individually_importable(self) -> None:
        import saccade

        module = importlib.import_module("saccade")
        for name in saccade.__all__:
            assert getattr(module, name) is not None

    def test_version_is_exposed(self) -> None:
        import saccade

        assert saccade.__version__ == "0.1.0"

    def test_public_modules_declare_all(self) -> None:
        """Public modules should say what they export."""
        public = [
            p for p in python_files() if not p.name.startswith("_") and p.name != "__init__.py"
        ]
        missing = [
            relative(p)
            for p in public
            if not any(
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "__all__"
                    for target in node.targets
                )
                for node in parse(p).body
            )
        ]
        assert not missing, f"public modules without __all__: {missing}"


class TestPackagingInvariants:
    def test_py_typed_marker_exists(self) -> None:
        """PEP 561: without this file, a user's mypy sees no types at all."""
        marker = SRC / "py.typed"
        assert marker.is_file()
        assert marker.stat().st_size == 0

    def test_internal_modules_are_underscore_prefixed(self) -> None:
        """The naming convention is the versioning contract (rule 4)."""
        internal_names = {"_planner.py", "_observer.py", "_verifier.py", "_evidence.py"}
        for path in python_files():
            if path.name in internal_names:
                assert path.name.startswith("_")
