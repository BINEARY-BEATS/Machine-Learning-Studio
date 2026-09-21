import pytest
from unittest.mock import MagicMock, patch
from ml_studio.gui.gallery_main import WidgetGalleryWindow, main
import sys

def test_gallery_main_init(qtbot):
    with patch("ml_studio.gui.gallery_main.apply_theme"):
        window = WidgetGalleryWindow()
        qtbot.addWidget(window)
        assert window.windowTitle() == "ML Studio — Design System Gallery (Slice 1)"

def test_gallery_main_toggle_theme(qtbot):
    with patch("ml_studio.gui.gallery_main.apply_theme"):
        window = WidgetGalleryWindow()
        qtbot.addWidget(window)
        assert window._theme_btn.text() == "Switch to Dark"
        
        window._toggle_theme()
        
        assert window._theme_btn.text() == "Switch to Light"

def test_gallery_main_show_toast(qtbot):
    with patch("ml_studio.gui.gallery_main.apply_theme"):
        window = WidgetGalleryWindow()
        qtbot.addWidget(window)
        with patch.object(window._toast, "show_message") as mock_show:
            window._show_toast("test", "success")
            mock_show.assert_called_once_with("test", variant="success")

def test_gallery_main_show_overlay(qtbot):
    with patch("ml_studio.gui.gallery_main.apply_theme"):
        window = WidgetGalleryWindow()
        qtbot.addWidget(window)
        with patch.object(window._overlay, "show_overlay") as mock_show:
            window._show_overlay_demo()
            mock_show.assert_called_once()

def test_gallery_main_main():
    with patch("ml_studio.gui.gallery_main.write_theme_files") as wtf:
        with patch("ml_studio.gui.gallery_main.apply_theme"):
            with patch("ml_studio.gui.gallery_main.QApplication.exec", return_value=0):
                # Since QApplication is a singleton in pytest-qt, just test it runs
                res = main()
                assert res == 0
                wtf.assert_called_once()
