"""Adapted from GeoSampler/PyniteGUI; see THIRD_PARTY.md for provenance."""
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


LIGHT = {
    "window": "#f5f7f7", "base": "#ffffff", "canvas": "#fafcfc",
    "alternate": "#f2f6f6", "text": "#24343b", "muted": "#69777e",
    "button": "#edf1f2", "panel": "#e5ebed", "border": "#bac6cb",
    "highlight": "#176b73", "highlight_text": "#ffffff", "selection": "#d5e8ea",
    "hover": "#e0eded", "checked_border": "#62979c", "shadow": "#47575c",
    "light": "#ffffff", "dark": "#86959c", "visited": "#495c91",
    "grid": "#e2e7e8", "axis": "#a8b5b8",
}
DARK = {
    "window": "#232629", "base": "#292d31", "canvas": "#191c1f",
    "alternate": "#30353a", "text": "#eef1f2", "muted": "#a1aab2",
    "button": "#353b40", "panel": "#30353a", "border": "#56616a",
    "highlight": "#236d73", "highlight_text": "#ffffff", "selection": "#334e52",
    "hover": "#3b4a4f", "checked_border": "#6dabae", "shadow": "#111416",
    "light": "#657079", "dark": "#15191c", "visited": "#bca4ed",
    "grid": "#30373b", "axis": "#58656c",
}
THEMES = {"light": LIGHT, "dark": DARK}


def theme_name():
    application = QApplication.instance()
    name = application.property("margam_theme") if application else None
    return name if name in THEMES else "light"


def colors():
    return THEMES[theme_name()]


def configure_theme(application, name="light"):
    name = name if isinstance(name, str) and name in THEMES else "light"
    if application.property("margam_theme") == name:
        return
    application.setProperty("margam_theme", name)
    application.setStyle("Fusion")
    c = THEMES[name]
    palette = QPalette()
    roles = {
        "Window": "window", "WindowText": "text", "Base": "base", "AlternateBase": "alternate",
        "Text": "text", "Button": "button", "ButtonText": "text", "Highlight": "highlight",
        "HighlightedText": "highlight_text", "ToolTipBase": "base", "ToolTipText": "text",
        "PlaceholderText": "muted", "Link": "highlight", "LinkVisited": "visited", "Light": "light",
        "Midlight": "panel", "Mid": "border", "Dark": "dark", "Shadow": "shadow",
        "BrightText": "highlight_text", "Accent": "highlight",
    }
    for role, key in roles.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(c[key]))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(c["button"]))
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Highlight, QColor(c["selection"]))
    palette.setColor(QPalette.ColorGroup.Inactive, QPalette.ColorRole.HighlightedText, QColor(c["text"]))
    application.setPalette(palette)
    application.setStyleSheet(f"""
        QMainWindow {{ background: {c['window']}; }}
        QToolBar {{ spacing: 4px; padding: 5px; background: {c['window']}; color: {c['text']}; border-bottom: 1px solid {c['border']}; }}
        QToolBar::separator {{ background: {c['border']}; width: 1px; margin: 5px 2px; }}
        QToolButton {{ padding: 4px; border: 1px solid transparent; border-radius: 3px; }}
        QToolButton:hover {{ background: {c['hover']}; }}
        QToolButton:checked {{ background: {c['selection']}; border-color: {c['checked_border']}; }}
        QDockWidget::title {{ padding: 7px; background: {c['panel']}; color: {c['text']}; }}
        QTreeWidget, QTableWidget {{ border: 0; alternate-background-color: {c['alternate']}; }}
        QPushButton {{ padding: 5px 10px; }}
        QStatusBar {{ background: {c['panel']}; color: {c['text']}; }}
    """)
    for widget in application.topLevelWidgets():
        if hasattr(widget, "apply_theme"):
            widget.apply_theme()

