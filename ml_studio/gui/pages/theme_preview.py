"""Design token swatch panel for the widget gallery."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ml_studio.app.theme import ThemeMode, color_token
from ml_studio.app.theme_tokens import RADIUS, SPACE, TYPE


class ThemePreviewPage(QWidget):
    """Scrollable preview of palette tokens and typography scale."""

    SWATCHES = (
        "background",
        "surface",
        "surface_raised",
        "border",
        "text",
        "text_muted",
        "primary",
        "success",
        "warning",
        "danger",
        "info",
    )

    def __init__(self, mode: ThemeMode = ThemeMode.LIGHT, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = mode
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE[5], SPACE[5], SPACE[5], SPACE[5])
        root.setSpacing(SPACE[4])

        title = QLabel("Design Tokens")
        title.setObjectName("PageTitle")
        root.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(SPACE[5])
        layout.addWidget(self._build_palette_section())
        layout.addWidget(self._build_type_section())
        layout.addWidget(self._build_radius_section())
        layout.addStretch()
        scroll.setWidget(body)
        root.addWidget(scroll)

    def set_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        # Rebuild is handled by gallery refresh on toggle.

    def _build_palette_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setSpacing(SPACE[3])
        heading = QLabel("Color Palette")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        grid = QGridLayout()
        grid.setSpacing(SPACE[3])
        for index, name in enumerate(self.SWATCHES):
            grid.addWidget(self._swatch(name), index // 4, index % 4)
        layout.addLayout(grid)
        return section

    def _swatch(self, token: str) -> QWidget:
        box = QFrame()
        box.setFixedSize(120, 72)
        hex_value = color_token(self._mode, token)
        box.setStyleSheet(
            f"background-color: {hex_value}; border: 1px solid {color_token(self._mode, 'border')};"
            f"border-radius: {RADIUS['md']}px;"
        )
        label = QLabel(f"{token}\n{hex_value}")
        label.setObjectName("SwatchLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer = QVBoxLayout()
        outer.addWidget(box)
        outer.addWidget(label)
        wrap = QWidget()
        wrap.setLayout(outer)
        return wrap

    def _build_type_section(self) -> QWidget:
        section = QWidget()
        layout = QVBoxLayout(section)
        heading = QLabel("Typography")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)
        for name, spec in TYPE.items():
            label = QLabel(f"{name} — The quick brown fox")
            size = int(spec["size"])
            weight = int(spec["weight"])
            label.setStyleSheet(f"font-size: {size}px; font-weight: {weight};")
            layout.addWidget(label)
        return section

    def _build_radius_section(self) -> QWidget:
        section = QWidget()
        row = QHBoxLayout(section)
        row.setSpacing(SPACE[4])
        heading = QLabel("Radius")
        heading.setObjectName("SectionTitle")
        row.addWidget(heading)
        for name, px in RADIUS.items():
            if name == "full":
                continue
            chip = QFrame()
            chip.setFixedSize(48, 48)
            chip.setStyleSheet(
                f"background-color: {color_token(self._mode, 'primary_subtle')};"
                f"border: 1px solid {color_token(self._mode, 'primary')};"
                f"border-radius: {px}px;"
            )
            row.addWidget(chip)
            row.addWidget(QLabel(name))
        row.addStretch()
        return section
