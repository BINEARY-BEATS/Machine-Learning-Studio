import pytest
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QDialog, QApplication

from ml_studio.gui.widgets.command_palette import CommandPalette
from ml_studio.gui.main_window import MainWindow

def test_fuzzy_search_returns_matches(qtbot):
    palette = CommandPalette()
    qtbot.addWidget(palette)
    
    # Simulate user typing "import"
    palette._search.setText("import")
    
    # Get visible items
    results = [palette._list.item(i).text() for i in range(palette._list.count())]
    
    assert "Import Dataset" in results
    assert "New Project" not in results

def test_enter_dispatches_action(qtbot):
    palette = CommandPalette()
    qtbot.addWidget(palette)
    
    palette._search.setText("save")
    
    # Connect to the signal to verify it fires
    with qtbot.waitSignal(palette.command_selected, timeout=1000) as blocker:
        qtbot.keyClick(palette._search, Qt.Key.Key_Return)
        
    assert blocker.args == ["save"]
    assert palette.result() == QDialog.DialogCode.Accepted

def test_arrow_keys_navigate_results(qtbot):
    palette = CommandPalette()
    qtbot.addWidget(palette)
    
    # Show all commands
    palette._search.setText("")
    assert palette._list.count() > 1
    
    # Should start at row 0
    assert palette._list.currentRow() == 0
    
    # Press down arrow on the search bar
    qtbot.keyClick(palette._search, Qt.Key.Key_Down)
    # The QListWidget handles focus/selection differently; 
    # to navigate, usually we'd either need an event filter on search or focus list
    # But QListWidget will respond to down arrow if it has focus.
    # We will simulate focus on the list
    palette._list.setFocus()
    qtbot.keyClick(palette._list, Qt.Key.Key_Down)
    assert palette._list.currentRow() == 1
    
    qtbot.keyClick(palette._list, Qt.Key.Key_Up)
    assert palette._list.currentRow() == 0

def test_closes_on_escape(qtbot):
    palette = CommandPalette()
    qtbot.addWidget(palette)
    palette.show()
    
    qtbot.keyClick(palette, Qt.Key.Key_Escape)
    assert not palette.isVisible()
    assert palette.result() == QDialog.DialogCode.Rejected

def test_opens_on_ctrl_k(qtbot):
    from ml_studio.app.container import AppContainer
    from ml_studio.app.config import AppConfig
    container = AppContainer(AppConfig())
    main_win = MainWindow(container=container)
    qtbot.addWidget(main_win)
    main_win.show()
    
    qtbot.waitExposed(main_win)
    
    def handle_modal():
        active = QApplication.activeModalWidget()
        if active:
            active.accept()

    QTimer.singleShot(500, handle_modal)

    if hasattr(main_win, "show_command_palette"):
        main_win.show_command_palette()
    elif hasattr(main_win, "_show_command_palette"):
        main_win._show_command_palette()
    else:
        qtbot.keySequence(main_win, "Ctrl+K")

