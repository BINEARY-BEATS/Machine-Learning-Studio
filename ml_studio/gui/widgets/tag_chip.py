"""Semantic tag chip labels."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget


class TagChip(QLabel):
    """Small pill label with semantic variant coloring via QSS."""

    VARIANTS = ("default", "success", "warning", "danger", "info")

    def __init__(self, text: str, variant: str = "default", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("TagChip")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_variant(variant)

    def set_variant(self, variant: str) -> None:
        safe = variant if variant in self.VARIANTS else "default"
        self.setProperty("chipVariant", safe)
        self.style().unpolish(self)
        self.style().polish(self)
