import pytest
from ml_studio.gui.widgets.schema_editor import SchemaEditor
from ml_studio.core.schema import ColumnSchema, ColumnKind, ColumnRole

@pytest.fixture
def columns():
    return [
        ColumnSchema(name="age", kind=ColumnKind.NUMERIC, dtype="int64", role=ColumnRole.FEATURE, sample_values=[25, 30, 35]),
        ColumnSchema(name="name", kind=ColumnKind.CATEGORICAL, dtype="object", role=ColumnRole.FEATURE, sample_values=["Alice", "Bob"])
    ]

def test_schema_editor_init(qtbot):
    editor = SchemaEditor()
    qtbot.addWidget(editor)
    assert editor._table.rowCount() == 0

def test_schema_editor_set_schema(qtbot, columns):
    editor = SchemaEditor()
    qtbot.addWidget(editor)
    editor.set_schema(columns, overrides={"name": "drop"})
    
    assert editor._table.rowCount() == 2
    assert editor._table.item(0, 0).text() == "age"
    assert editor._table.item(1, 0).text() == "name"
    
    # Check overrides
    combo = editor._table.cellWidget(1, 3)
    assert combo.currentData() == "drop"

def test_schema_editor_role_changed(qtbot, columns):
    editor = SchemaEditor()
    qtbot.addWidget(editor)
    editor.set_schema(columns)
    
    combo = editor._table.cellWidget(0, 3)
    
    with qtbot.waitSignal(editor.role_changed, timeout=1000) as blocker:
        idx = combo.findData("target")
        combo.setCurrentIndex(idx)
        
    assert blocker.args == ["age", "target"]
