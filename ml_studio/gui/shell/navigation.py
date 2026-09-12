"""Application shell navigation definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NavItem:
    key: str
    label: str
    icon: str
    section: str


NAV_SECTIONS: tuple[str, ...] = ("Workspace", "Analysis", "Deploy")

NAV_ITEMS: tuple[NavItem, ...] = (
    NavItem("home", "Home", "home", "Workspace"),
    NavItem("data", "Data", "table", "Workspace"),
    NavItem("prepare", "Prepare", "clean", "Workspace"),
    NavItem("train", "Train", "train", "Analysis"),
    NavItem("evaluate", "Evaluate", "chart", "Analysis"),
    NavItem("predict", "Predict", "ai", "Deploy"),
    NavItem("models", "Models", "clipboard", "Deploy"),
    NavItem("settings", "Settings", "context", "Deploy"),
)

PAGE_KEYS: tuple[str, ...] = tuple(item.key for item in NAV_ITEMS)
