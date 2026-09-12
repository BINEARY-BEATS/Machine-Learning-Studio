"""Pre-commit style checks for Slice 1+ coding rules."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ML_STUDIO = ROOT / "ml_studio"
ALLOWED_HEX_FILES = {
    ML_STUDIO / "app" / "theme_tokens.py",
    ML_STUDIO / "assets" / "themes" / "light.qss",
    ML_STUDIO / "assets" / "themes" / "dark.qss",
}
SLICE1_FILES = {
    ML_STUDIO / "app" / "theme.py",
    ML_STUDIO / "app" / "theme_tokens.py",
    ML_STUDIO / "app" / "theme_qss.py",
    ML_STUDIO / "app" / "metric_color.py",
    ML_STUDIO / "app" / "icon_provider.py",
    ML_STUDIO / "gui" / "gallery_main.py",
    ML_STUDIO / "gui" / "pages" / "theme_preview.py",
    ML_STUDIO / "gui" / "widgets" / "card.py",
    ML_STUDIO / "gui" / "widgets" / "stat_card.py",
    ML_STUDIO / "gui" / "widgets" / "icon_button.py",
    ML_STUDIO / "gui" / "widgets" / "search_bar.py",
    ML_STUDIO / "gui" / "widgets" / "tag_chip.py",
    ML_STUDIO / "gui" / "widgets" / "empty_state.py",
    ML_STUDIO / "gui" / "widgets" / "loading_overlay.py",
    ML_STUDIO / "gui" / "widgets" / "toast.py",
}
HEX_PATTERN = re.compile(r"#[0-9A-Fa-f]{3,8}\b")
SKIP_DIRS = {"tests", "__pycache__", ".pytest_cache"}
MAX_FILE_LINES = 400
MAX_FUNC_LINES = 40
MAX_CLASS_LINES = 300
BANNED = re.compile(r"\b(TODO|FIXME)\b|#\s*placeholder\b|\bpass\s+#")


def main() -> int:
    errors: list[str] = []
    errors.extend(_check_hex_outside_theme())
    errors.extend(_check_line_limits())
    errors.extend(_check_banned_tokens())
    if errors:
        for err in errors:
            print(err, file=sys.stderr)
        print(f"\n{len(errors)} rule violation(s).", file=sys.stderr)
        return 1
    print("All rule checks passed.")
    return 0


def _slice1_files() -> list[Path]:
    return sorted(SLICE1_FILES)


def _check_hex_outside_theme() -> list[str]:
    errors: list[str] = []
    for path in _slice1_files():
        if path in ALLOWED_HEX_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        for match in HEX_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            errors.append(f"HEX outside theme: {path.relative_to(ROOT)}:{line} -> {match.group()}")
    return errors


def _check_line_limits() -> list[str]:
    errors: list[str] = []
    for path in _slice1_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_FILE_LINES:
            errors.append(f"File > {MAX_FILE_LINES} lines: {path.relative_to(ROOT)} ({len(lines)})")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                span = _node_span(node)
                if span > MAX_FUNC_LINES:
                    errors.append(
                        f"Function > {MAX_FUNC_LINES} lines: {path.relative_to(ROOT)}:"
                        f"{node.name} ({span})"
                    )
            if isinstance(node, ast.ClassDef):
                span = _node_span(node)
                if span > MAX_CLASS_LINES:
                    errors.append(
                        f"Class > {MAX_CLASS_LINES} lines: {path.relative_to(ROOT)}:"
                        f"{node.name} ({span})"
                    )
    return errors


def _node_span(node: ast.AST) -> int:
    if not hasattr(node, "end_lineno") or not hasattr(node, "lineno"):
        return 0
    return int(node.end_lineno) - int(node.lineno) + 1


def _check_banned_tokens() -> list[str]:
    errors: list[str] = []
    for path in _slice1_files():
        if path.parent.name != "widgets":
            continue
        text = path.read_text(encoding="utf-8")
        if BANNED.search(text):
            errors.append(f"Banned token in {path.relative_to(ROOT)}")
    return errors


if __name__ == "__main__":
    sys.exit(main())
