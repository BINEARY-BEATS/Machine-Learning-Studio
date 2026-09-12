"""Transform base classes and registry."""

from __future__ import annotations

from typing import Type

from ml_studio.core.pipeline import TransformStep

TRANSFORM_REGISTRY: dict[str, Type[TransformStep]] = {}


def register_transform(name: str):
    def decorator(cls: Type[TransformStep]) -> Type[TransformStep]:
        TRANSFORM_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_transform(name: str, **params) -> TransformStep:
    if name not in TRANSFORM_REGISTRY:
        raise KeyError(f"Unknown transform: {name}")
    return TRANSFORM_REGISTRY[name](**params)


def list_transforms() -> list[str]:
    return sorted(TRANSFORM_REGISTRY.keys())
