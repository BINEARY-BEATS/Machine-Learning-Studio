"""Shared layout helpers for responsive GUI pages."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHeaderView,
    QLayout,
    QPushButton,
    QSizePolicy,
    QWidget,
)


def fill_widget(layout: QLayout, widget: QWidget, stretch: int = 1) -> None:
    """Add a widget that expands to fill available vertical/horizontal space."""
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    layout.addWidget(widget, stretch)


def constrain_primary_button(btn: QPushButton) -> None:
    """Prevent primary CTAs from stretching full card width."""
    btn.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    btn.setMinimumWidth(120)


def configure_table_header(
    header: QHeaderView,
    *,
    contents_cols: tuple[int, ...] = (),
    stretch_cols: tuple[int, ...] = (),
    interactive_rest: bool = True,
) -> None:
    """Set column resize modes: contents for key cols, stretch for detail cols.

    Remaining columns become Interactive when interactive_rest is True.
    """
    count = header.count()
    if count <= 0:
        return

    default = (
        QHeaderView.ResizeMode.Interactive
        if interactive_rest
        else QHeaderView.ResizeMode.ResizeToContents
    )
    for i in range(count):
        header.setSectionResizeMode(i, default)

    for i in contents_cols:
        if 0 <= i < count:
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

    stretch = list(stretch_cols)
    if not stretch and count > 0:
        stretch = [count - 1]
    for i in stretch:
        if 0 <= i < count:
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

    header.setStretchLastSection(True)
    header.setMinimumSectionSize(48)
