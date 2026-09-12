"""Custom Python Transform."""

import pandas as pd
from .base import BaseTransform


class CustomPython(BaseTransform):
    """Custom Python transform (Stub for Phase 4)."""

    def __init__(self, code=""):
        super().__init__(code=code)
        self.code = code

    @classmethod
    def get_schema(cls) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "default": ""}
            }
        }

    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        raise NotImplementedError("CustomPython transform is deferred to Phase 4.")

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("CustomPython transform is deferred to Phase 4.")

    def to_dict(self) -> dict:
        return self.params.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        return cls(code=d.get("code", ""))
