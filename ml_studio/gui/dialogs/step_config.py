import json
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, 
                            QComboBox, QPushButton, QFormLayout, QWidget, QScrollArea, QListWidget, QListWidgetItem, QGroupBox)
from PyQt6.QtCore import Qt

from ml_studio.transforms.registry import get as get_transform

class MultiSelectWidget(QWidget):
    def __init__(self, available_options, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget()
        self._list.setMaximumHeight(100)
        self._layout.addWidget(self._list)
        
        for opt in available_options:
            item = QListWidgetItem(opt)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._list.addItem(item)
            
    def set_value(self, values):
        if not values:
            return
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.text() in values:
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setCheckState(Qt.CheckState.Unchecked)
                
    def get_value(self):
        values = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                values.append(item.text())
        return values

class StepConfigDialog(QDialog):
    def __init__(self, transform_name: str, current_params: dict, columns: list[str], parent=None):
        super().__init__(parent)
        self.transform_class = get_transform(transform_name)
        self.current_params = current_params or {}
        self.available_columns = columns
        
        self.schema = self.transform_class.get_schema()
        self.final_params = None
        self._fields = {}
        
        self.setWindowTitle(f"Configure {transform_name}")
        self.setMinimumWidth(400)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self._form = QFormLayout(content)
        
        props = self.schema.get("properties", {})
        for name, prop in props.items():
            widget = self._create_field(name, prop)
            if widget:
                self._fields[name] = widget
                # special case for QCheckBox to align differently if wanted, but form layout is fine
                self._form.addRow(name, widget)
                
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        self._error_label = QLabel("")
        self._error_label.setObjectName("DangerText")
        self._error_label.setStyleSheet("color: red;")
        self._error_label.hide()
        layout.addWidget(self._error_label)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_layout.addWidget(cancel)
        
        save = QPushButton("Save")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self._on_save)
        btn_layout.addWidget(save)
        
        layout.addLayout(btn_layout)

    def _create_field(self, name: str, prop: dict):
        ptype = prop.get("type")
        default_val = self.current_params.get(name, prop.get("default"))
        
        if ptype == "integer":
            w = QSpinBox()
            w.setMinimum(prop.get("minimum", -999999))
            w.setMaximum(prop.get("maximum", 999999))
            if default_val is not None:
                w.setValue(int(default_val))
            return w
            
        elif ptype == "number":
            w = QDoubleSpinBox()
            w.setMinimum(prop.get("minimum", -999999.0))
            w.setMaximum(prop.get("maximum", 999999.0))
            w.setDecimals(4)
            if default_val is not None:
                w.setValue(float(default_val))
            return w
            
        elif ptype == "boolean":
            w = QCheckBox()
            if default_val is not None:
                w.setChecked(bool(default_val))
            return w
            
        elif ptype == "string" and "enum" in prop:
            w = QComboBox()
            w.addItems([str(x) for x in prop["enum"]])
            if default_val is not None:
                w.setCurrentText(str(default_val))
            return w
            
        elif ptype == "string" or ptype == ["number", "string"]:
            w = QLineEdit()
            if default_val is not None:
                w.setText(str(default_val))
            return w
            
        elif ptype == "array":
            items_type = prop.get("items", {}).get("type")
            if items_type == "string" and name == "columns":
                w = MultiSelectWidget(self.available_columns)
                if default_val is not None:
                    w.set_value(default_val)
                return w
            else:
                # Fallback for array of numbers/tuples
                w = QLineEdit()
                w.setPlaceholderText("JSON format e.g. [1, 2]")
                if default_val is not None:
                    w.setText(json.dumps(default_val))
                return w
                
        elif ptype == "object":
            w = QLineEdit()
            w.setPlaceholderText("JSON format e.g. {\"A\": 1}")
            if default_val is not None:
                w.setText(json.dumps(default_val))
            return w
            
        return QLineEdit()

    def _get_values(self):
        result = {}
        props = self.schema.get("properties", {})
        for name, widget in self._fields.items():
            ptype = props.get(name, {}).get("type")
            
            if isinstance(widget, QSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                result[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                result[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                result[name] = widget.currentText()
            elif isinstance(widget, MultiSelectWidget):
                result[name] = widget.get_value()
            elif isinstance(widget, QLineEdit):
                text = widget.text().strip()
                if not text:
                    continue
                if ptype == "array" or ptype == "object":
                    try:
                        result[name] = json.loads(text)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON for {name}")
                elif ptype == ["number", "string"]:
                    # Try to parse as float if possible
                    try:
                        if "." in text:
                            result[name] = float(text)
                        else:
                            result[name] = int(text)
                    except ValueError:
                        result[name] = text
                else:
                    result[name] = text
        return result

    def _on_save(self):
        self._error_label.hide()
        try:
            params = self._get_values()
            # Validation Step R2: Ensure params are valid for the schema
            transform = self.transform_class(**params)
            transform.get_schema()  # Implicitly triggers pydantic/dataclass validation if any
            
            self.final_params = params
            self.accept()
        except Exception as e:
            self._error_label.setText(str(e))
            self._error_label.show()

