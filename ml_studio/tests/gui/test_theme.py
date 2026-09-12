"""Theme token and QSS tests."""

from __future__ import annotations

from ml_studio.app.theme import (
    MOTION,
    RADIUS,
    SPACE,
    TYPE,
    ThemeMode,
    apply_theme,
    build_stylesheet,
    color_token,
    palette_for,
    write_theme_files,
)


def test_space_scale_is_monotonic():
    values = [SPACE[i] for i in sorted(SPACE)]
    assert values == sorted(values)
    assert SPACE[0] == 0


def test_type_tokens_have_required_keys():
    for name, spec in TYPE.items():
        assert "size" in spec
        assert "weight" in spec


def test_palette_has_semantic_colors():
    for mode in ThemeMode:
        p = palette_for(mode)
        assert p.primary
        assert p.danger
        assert p.metric_na


def test_build_stylesheet_contains_object_names():
    qss = build_stylesheet(ThemeMode.LIGHT)
    for name in ("#Card", "#SearchBar", "#TagChip", "#LoadingOverlay", "#ToastLabel"):
        assert name in qss


def test_write_theme_files(tmp_path, monkeypatch):
    monkeypatch.setattr("ml_studio.app.theme.THEMES_DIR", tmp_path)
    light, dark = write_theme_files()
    assert light.exists()
    assert dark.exists()
    assert "#Card" in light.read_text(encoding="utf-8")


def test_apply_theme_sets_mode_property(qapp):
    from ml_studio.app.theme import apply_theme

    apply_theme(qapp, ThemeMode.DARK)
    assert qapp.property("themeMode") == "dark"


def test_color_token_roundtrip():
    assert color_token(ThemeMode.LIGHT, "primary") == palette_for(ThemeMode.LIGHT).primary
