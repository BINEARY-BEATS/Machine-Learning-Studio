"""Base Transform Interface."""

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseTransform(BaseEstimator, TransformerMixin, ABC):
    """Every transform implements this. No exceptions."""

    def __init__(self, **kwargs):
        self.params = kwargs
        self.fitted_columns_ = []
        self._is_fitted = False

    def get_params(self, deep=True):
        """Return parameters for sklearn compatibility."""
        return self.params

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        self.params.update(params)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @classmethod
    @abstractmethod
    def get_schema(cls) -> dict:
        """Return JSON schema for parameters. Used by GUI + CLI + validation."""
        pass

    def validate_input(self, X: pd.DataFrame) -> None:
        """Raise ValueError if X is not compatible with this transform."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError(f"{self.__class__.__name__} expects a pandas DataFrame, got {type(X)}.")
        
        # Base implementation checks if fitted_columns_ exist in X during transform
        if self._is_fitted:
            missing = [col for col in self.fitted_columns_ if col not in X.columns]
            if missing:
                raise KeyError(f"{self.__class__.__name__} missing fitted columns in input: {missing}")

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BaseTransform":
        """Learn parameters from training data ONLY."""
        self.validate_input(X)
        self._fit(X, y)
        self._is_fitted = True
        return self

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        """Internal fit implementation."""
        pass

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned transformation. No fitting here."""
        if not self._is_fitted:
            raise ValueError(f"This {self.__class__.__name__} instance is not fitted yet.")
        self.validate_input(X)
        
        # Make a copy of X to avoid mutating the original
        X_out = X.copy()
        
        # Apply transformation only on fitted columns
        return self._transform(X_out)

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Internal transform implementation."""
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize parameters + learned state to JSON-compatible dict."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "BaseTransform":
        """Reconstruct from dict."""
        pass

    def get_output_columns(self, input_columns: list[str]) -> list[str]:
        """Predict output column names. Used by pipeline preview."""
        return input_columns
