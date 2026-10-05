"""Schema-driven prediction input widgets (no Qt in core)."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

MISSING = "(missing)"


def schema_or_empty(pipeline) -> dict[str, dict]:
    raw = getattr(pipeline, "feature_schema", None) if pipeline is not None else None
    return raw if isinstance(raw, dict) else {}


def clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()
        child = item.layout()
        if child is not None:
            clear_layout(child)


def build_feature_row(
    col: str, spec: dict | None, parent: QWidget
) -> tuple[QWidget, QWidget, QLabel]:
    """Return (row_widget, input_widget, error_label)."""
    kind = (spec or {}).get("kind", "")
    label = QLabel(f"{col}  ·  {kind or 'raw'}")
    label.setObjectName("TextMuted")
    inp = _make_input(kind, spec or {})
    err = QLabel("")
    err.setObjectName("ValidationError")
    err.hide()
    col_box = QVBoxLayout()
    col_box.setContentsMargins(0, 0, 0, 0)
    col_box.setSpacing(2)
    col_box.addWidget(label)
    col_box.addWidget(inp)
    col_box.addWidget(err)
    wrap = QWidget(parent)
    wrap.setLayout(col_box)
    return wrap, inp, err


def _make_input(kind: str, spec: dict) -> QWidget:
    if kind == "numeric":
        inp = QLineEdit()
        inp.setValidator(QDoubleValidator())
        lo, hi, ref = spec.get("min"), spec.get("max"), spec.get("ref")
        bits = []
        if lo is not None and hi is not None:
            bits.append(f"{lo:g}–{hi:g}")
        if ref is not None:
            bits.append(f"ref {ref:g}")
        inp.setPlaceholderText(" · ".join(bits) if bits else "number (blank = missing)")
        return inp
    if kind == "categorical":
        box = QComboBox()
        box.setEditable(True)
        box.addItem(MISSING)
        for cat in spec.get("categories") or []:
            box.addItem(str(cat))
        box.setCurrentIndex(0)
        return box
    if kind == "boolean":
        cb = QCheckBox("true / false (partial = missing)")
        cb.setTristate(True)
        cb.setCheckState(Qt.CheckState.PartiallyChecked)
        return cb
    if kind == "datetime":
        inp = QLineEdit()
        inp.setPlaceholderText("ISO datetime (blank = missing)")
        return inp
    return QLineEdit()


def collect_values(inputs: dict[str, QWidget]) -> dict[str, Any]:
    return {col: _widget_value(w) for col, w in inputs.items()}


def _widget_value(w: QWidget) -> Any:
    if isinstance(w, QComboBox):
        text = w.currentText().strip()
        return "" if text == MISSING else text
    if isinstance(w, QCheckBox):
        state = w.checkState()
        if state == Qt.CheckState.PartiallyChecked:
            return ""
        return state == Qt.CheckState.Checked
    if isinstance(w, QLineEdit):
        return w.text()
    return ""


def clear_field_errors(inputs: dict[str, QWidget], error_labels: dict[str, QLabel]) -> None:
    for col, w in inputs.items():
        _set_error(w, error_labels.get(col), False, "")


def mark_field_errors(
    inputs: dict[str, QWidget],
    error_labels: dict[str, QLabel],
    errors: dict[str, str],
) -> None:
    clear_field_errors(inputs, error_labels)
    for col, msg in errors.items():
        w = inputs.get(col)
        if w is not None:
            _set_error(w, error_labels.get(col), True, msg)


def _set_error(w: QWidget, err_label: QLabel | None, has_error: bool, message: str) -> None:
    w.setProperty("error", has_error)
    w.setToolTip(message if has_error else "")
    style = w.style()
    style.unpolish(w)
    style.polish(w)
    if err_label is not None:
        if has_error:
            err_label.setText(message)
            err_label.show()
        else:
            err_label.hide()
            err_label.clear()


def render_proba_bars(host_layout, probabilities: dict[str, float] | None) -> None:
    clear_layout(host_layout)
    if not probabilities:
        return
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for label, prob in ranked:
        row = QHBoxLayout()
        name = QLabel(str(label))
        name.setMinimumWidth(72)
        bar = QProgressBar()
        bar.setObjectName("ClassProbBar")
        bar.setRange(0, 100)
        bar.setValue(int(round(float(prob) * 100)))
        bar.setFormat(f"{float(prob):.1%}")
        bar.setTextVisible(True)
        row.addWidget(name)
        row.addWidget(bar, 1)
        wrap = QWidget()
        wrap.setLayout(row)
        host_layout.addWidget(wrap)
