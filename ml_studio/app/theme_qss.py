"""Global QSS generation from design tokens."""

from __future__ import annotations

from ml_studio.app.theme_tokens import (
    FONT_SANS,
    RADIUS,
    SPACE,
    ThemeMode,
    palette_for,
)


def build_stylesheet(mode: ThemeMode) -> str:
    """Build complete application QSS for the given theme mode."""
    p = palette_for(mode)
    s, r = SPACE, RADIUS
    sections = [
        _base_qss(p, s, r),
        _primary_buttons_qss(p, s, r),
        _icon_buttons_qss(p, s, r),
        _inputs_qss(p, s, r),
        _tables_qss(p, s, r),
        _card_qss(p, s, r),
        _stat_qss(p),
        _empty_toast_qss(p, s, r),
        _tabs_qss(p, s, r),
        _tag_chip_qss(p, s, r),
        _overlay_qss(p, s, r),
    ]
    return "\n".join(sections)


def _base_qss(p, s, r) -> str:
    return f"""
    QWidget {{
        font-family: {FONT_SANS};
        font-size: 13px;
        color: {p.text};
    }}
    QMainWindow, #CentralWidget, #GalleryRoot {{
        background-color: {p.background};
    }}
    #Sidebar {{
        background-color: {p.surface};
        border-right: 1px solid {p.border};
    }}
    #ContentArea {{ background-color: {p.background}; }}
    #StatusBar {{
        background-color: {p.surface};
        border-top: 1px solid {p.border};
        color: {p.text_muted};
        padding: {s[1]}px {s[3]}px;
    }}
    #PageTitle {{ font-size: 18px; font-weight: 600; color: {p.text}; }}
    #SectionTitle {{ font-size: 14px; font-weight: 600; color: {p.text}; }}
    """


def _primary_buttons_qss(p, s, r) -> str:
    return f"""
    QPushButton#PrimaryButton {{
        background-color: {p.primary};
        color: {p.on_primary};
        border: none;
        border-radius: {r['md']}px;
        padding: {s[2]}px {s[4]}px;
        font-weight: 600;
    }}
    QPushButton#PrimaryButton:hover {{ background-color: {p.primary_hover}; }}
    QPushButton#PrimaryButton:pressed {{ background-color: {p.primary_hover}; }}
    QPushButton#PrimaryButton:disabled {{
        background-color: {p.border};
        color: {p.text_disabled};
    }}
    QPushButton#PrimaryButton:focus {{
        outline: 2px solid {p.primary};
        outline-offset: 2px;
    }}
    """


def _icon_buttons_qss(p, s, r) -> str:
    return f"""
    QPushButton#IconButton {{
        background: transparent;
        border: 1px solid transparent;
        border-radius: {r['md']}px;
        padding: {s[2]}px;
        color: {p.text_muted};
    }}
    QPushButton#IconButton:hover {{
        background-color: {p.surface_raised};
        color: {p.text};
        border-color: {p.border_subtle};
    }}
    QPushButton#IconButton:pressed {{ background-color: {p.border_subtle}; }}
    QPushButton#IconButton:disabled {{ color: {p.text_disabled}; }}
    QPushButton#IconButton:focus {{ border-color: {p.primary}; }}
    QPushButton#GhostButton {{
        background: transparent;
        border: 1px solid {p.border};
        border-radius: {r['md']}px;
        padding: {s[2]}px {s[4]}px;
        color: {p.text};
    }}
    QPushButton#GhostButton:hover {{ background-color: {p.surface_raised}; }}
    """


def _inputs_qss(p, s, r) -> str:
    return f"""
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r['sm']}px;
        padding: {s[2]}px {s[3]}px;
        color: {p.text};
        selection-background-color: {p.primary_subtle};
    }}
    QLineEdit:focus, QComboBox:focus {{ border-color: {p.primary}; }}
    QLineEdit#SearchBar {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_subtle};
        border-radius: {r['full']}px;
        padding: {s[2]}px {s[4]}px;
        padding-left: {s[6]}px;
    }}
    QLineEdit#SearchBar:focus {{
        border-color: {p.primary};
        background-color: {p.surface};
    }}
    QComboBox QAbstractItemView {{
        background-color: {p.surface};
        color: {p.text};
        selection-background-color: {p.primary};
        selection-color: {p.on_primary};
    }}
    """


def _tables_qss(p, s, r) -> str:
    return f"""
    QTableView, QTableWidget {{
        background-color: {p.surface};
        alternate-background-color: {p.surface_raised};
        border: 1px solid {p.border};
        gridline-color: {p.border_subtle};
        color: {p.text};
    }}
    QHeaderView::section {{
        background-color: {p.surface_raised};
        border: none;
        border-bottom: 1px solid {p.border};
        padding: {s[2]}px;
        font-weight: 600;
        color: {p.text};
    }}
    QScrollBar:vertical {{
        background: {p.surface_raised};
        width: 10px;
        border-radius: {r['sm']}px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border};
        border-radius: {r['sm']}px;
        min-height: 24px;
    }}
    """


def _card_qss(p, s, r) -> str:
    return f"""
    #Card {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r['lg']}px;
    }}
    #CardHeader {{
        color: {p.text_muted};
        font-size: 12px;
        font-weight: 600;
    }}
    """


def _stat_qss(p) -> str:
    return f"""
    QLabel#StatTitle {{ color: {p.text_muted}; font-size: 12px; }}
    QLabel#StatValue {{ color: {p.text}; font-size: 22px; font-weight: 600; }}
    QLabel#StatDelta {{ font-size: 11px; font-weight: 500; }}
    """


def _empty_toast_qss(p, s, r) -> str:
    return f"""
    #EmptyStateTitle {{ font-size: 16px; font-weight: 600; color: {p.text}; }}
    #EmptyStateDescription {{ font-size: 13px; color: {p.text_muted}; }}
    #ToastLabel {{
        background-color: {p.surface_raised};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {r['md']}px;
        padding: {s[3]}px {s[4]}px;
        font-size: 13px;
    }}
    #ToastLabel[toastVariant="success"] {{ border-color: {p.success}; }}
    #ToastLabel[toastVariant="danger"] {{ border-color: {p.danger}; }}
    #ToastLabel[toastVariant="warning"] {{ border-color: {p.warning}; }}
    #LoadingMessage {{
        color: {p.text};
        font-size: 14px;
        font-weight: 600;
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r['md']}px;
        padding: {s[4]}px {s[5]}px;
    }}
    #SwatchLabel {{ font-size: 11px; color: {p.text_muted}; }}
    """


def _tabs_qss(p, s, r) -> str:
    return f"""
    QTabWidget::pane {{
        background-color: {p.surface};
        border: 1px solid {p.border};
        border-radius: {r['md']}px;
    }}
    QTabBar::tab {{
        background-color: {p.surface_raised};
        color: {p.text_muted};
        border: 1px solid {p.border};
        padding: {s[2]}px {s[4]}px;
        border-top-left-radius: {r['sm']}px;
        border-top-right-radius: {r['sm']}px;
    }}
    QTabBar::tab:selected {{
        background-color: {p.surface};
        color: {p.text};
        font-weight: 600;
    }}
    """


def _tag_chip_qss(p, s, r) -> str:
    base = f"""
    QLabel#TagChip {{
        border-radius: {r['full']}px;
        padding: {s[1]}px {s[3]}px;
        font-size: 11px;
        font-weight: 600;
        background-color: {p.surface_raised};
        color: {p.text_muted};
        border: 1px solid {p.border_subtle};
    }}
    """
    variants = {
        "success": (p.success, p.primary_subtle),
        "warning": (p.warning, p.surface_raised),
        "danger": (p.danger, p.surface_raised),
        "info": (p.info, p.primary_subtle),
    }
    extra = ""
    for name, (fg, bg) in variants.items():
        extra += f"""
    QLabel#TagChip[chipVariant="{name}"] {{
        color: {fg};
        background-color: {bg};
        border-color: {fg};
    }}
    """
    return base + extra


def _overlay_qss(p, s, r) -> str:
    return f"""
    #LoadingOverlay {{ background-color: {p.scrim}; }}
    #TopBar {{
        background-color: {p.surface};
        border-bottom: 1px solid {p.border};
    }}
    #Breadcrumb {{ color: {p.text_muted}; font-size: 12px; }}
    #ProjectName {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_subtle};
        border-radius: {r['sm']}px;
        padding: {s[1]}px {s[2]}px;
    }}
    #NavSection {{
        color: {p.text_muted};
        font-size: 10px;
        font-weight: 700;
        padding: {s[2]}px {s[3]}px {s[1]}px;
    }}
    #StatusChip {{
        color: {p.text_muted};
        font-size: 11px;
        padding-left: {s[3]}px;
    }}
    #ErrorBanner {{
        color: {p.danger};
        background-color: {p.surface_raised};
        border: 1px solid {p.danger};
        border-radius: {r['md']}px;
        padding: {s[3]}px;
    }}
    QPushButton#PillTab {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_subtle};
        border-radius: {r['full']}px;
        padding: {s[2]}px {s[4]}px;
        color: {p.text_muted};
    }}
    QPushButton#PillTab[active="true"] {{
        background-color: {p.primary_subtle};
        border-color: {p.primary};
        color: {p.primary};
        font-weight: 600;
    }}
    QPushButton#StepButton {{
        background-color: {p.surface_raised};
        border: 1px solid {p.border_subtle};
        border-radius: {r['md']}px;
        padding: {s[1]}px {s[3]}px;
        color: {p.text_muted};
        font-size: 11px;
    }}
    QPushButton#StepButton[stepState="active"] {{
        background-color: {p.primary};
        color: {p.on_primary};
        border-color: {p.primary};
    }}
    QPushButton#StepButton[stepState="done"] {{
        border-color: {p.success};
        color: {p.success};
    }}
    #NavButton {{
        text-align: left;
        padding: {s[2]}px {s[3]}px;
        border: none;
        border-radius: {r['md']}px;
        background: transparent;
        color: {p.text_muted};
    }}
    #NavButton:hover {{
        background-color: {p.surface_raised};
        color: {p.text};
    }}
    #NavButton[active="true"] {{
        background-color: {p.primary};
        color: {p.on_primary};
    }}
    QDialog, QListWidget {{
        background-color: {p.surface};
        color: {p.text};
        border: 1px solid {p.border};
        border-radius: {r['md']}px;
    }}
    """
