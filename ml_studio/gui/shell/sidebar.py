"""Collapsible sidebar with themed icons and section groups."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, pyqtSignal
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from ml_studio.app.icon_provider import themed_icon
from ml_studio.app.theme import ThemeMode
from ml_studio.app.theme_tokens import MOTION, SPACE
from ml_studio.gui.shell.navigation import NAV_ITEMS


class Sidebar(QFrame):
    """Left navigation rail with collapse animation."""

    navigate = pyqtSignal(str)
    EXPANDED_WIDTH = 220
    COLLAPSED_WIDTH = 64

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self._mode = ThemeMode.LIGHT
        self._collapsed = False
        self._buttons: dict[str, QPushButton] = {}
        self.setFixedWidth(self.EXPANDED_WIDTH)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE[2], SPACE[3], SPACE[2], SPACE[3])
        outer.setSpacing(SPACE[2])

        header = QHBoxLayout()
        self._title = QLabel("ML Studio")
        self._title.setObjectName("SidebarTitle")
        self._collapse_btn = QPushButton("«")
        self._collapse_btn.setObjectName("IconButton")
        self._collapse_btn.setToolTip("Collapse sidebar")
        self._collapse_btn.clicked.connect(self.toggle_collapse)
        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._collapse_btn)
        outer.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        self._body_layout = QVBoxLayout(body)
        self._body_layout.setSpacing(SPACE[1])
        self._build_nav()
        self._body_layout.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

    def _build_nav(self) -> None:
        current_section = ""
        for item in NAV_ITEMS:
            if item.section != current_section:
                current_section = item.section
                section_label = QLabel(current_section.upper())
                section_label.setObjectName("NavSection")
                self._body_layout.addWidget(section_label)
            btn = QPushButton(item.label)
            btn.setObjectName("NavButton")
            btn.setToolTip(item.label)
            btn.clicked.connect(lambda _checked, key=item.key: self._on_nav(key))
            self._buttons[item.key] = btn
            self._body_layout.addWidget(btn)
        self._apply_icons()

    def set_theme_mode(self, mode: ThemeMode) -> None:
        self._mode = mode
        self._apply_icons()

    def _apply_icons(self) -> None:
        for item in NAV_ITEMS:
            btn = self._buttons[item.key]
            icon = themed_icon(item.icon, self._mode.value, "text_muted")
            btn.setIcon(icon)

    def set_active(self, key: str) -> None:
        for nav_key, btn in self._buttons.items():
            btn.setProperty("active", "true" if nav_key == key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_nav(self, key: str) -> None:
        self.set_active(key)
        self.navigate.emit(key)

    def toggle_collapse(self) -> None:
        self._collapsed = not self._collapsed
        target = self.COLLAPSED_WIDTH if self._collapsed else self.EXPANDED_WIDTH
        self._collapse_btn.setText("»" if self._collapsed else "«")
        self._title.setVisible(not self._collapsed)
        for item in NAV_ITEMS:
            self._buttons[item.key].setText("" if self._collapsed else item.label)
        anim = QPropertyAnimation(self, b"minimumWidth", self)
        anim.setDuration(MOTION["normal"])
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        anim.start()
        self.setMaximumWidth(target)
        self.setFixedWidth(target)
