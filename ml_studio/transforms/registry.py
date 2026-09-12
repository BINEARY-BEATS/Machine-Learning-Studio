"""Transform Registry for auto-discovery."""

import importlib
import pkgutil
from typing import Type

from .base import BaseTransform

# Globals
_REGISTRY: dict[str, Type[BaseTransform]] = {}
_CATEGORIES: dict[str, list[Type[BaseTransform]]] = {}

def _auto_discover():
    """Discover all BaseTransform subclasses in the ml_studio.transforms package."""
    if _REGISTRY:
        return

    # Import all submodules to trigger class registration
    import ml_studio.transforms
    for _, module_name, _ in pkgutil.iter_modules(ml_studio.transforms.__path__):
        if module_name not in ["base", "registry"]:
            importlib.import_module(f"ml_studio.transforms.{module_name}")


    def get_all_subclasses(cls):
        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))
        return all_subclasses

    # Build registry
    for cls in get_all_subclasses(BaseTransform):
        # Skip base classes that aren't meant to be instantiated directly
        if cls.__name__ in ['BaseKFoldEncoder']:
            continue
        name = cls.__name__
        _REGISTRY[name] = cls

        
        # Determine category based on module name
        module_name = cls.__module__.split('.')[-1]
        if module_name not in _CATEGORIES:
            _CATEGORIES[module_name] = []
        _CATEGORIES[module_name].append(cls)


def list_all() -> list[dict]:
    """Return a list of all available transforms."""
    _auto_discover()
    result = []
    for category, classes in _CATEGORIES.items():
        for cls in classes:
            result.append({
                "name": cls.__name__,
                "category": category,
                "schema": cls.get_schema(),
                "doc": cls.__doc__
            })
    return result

def get(name: str) -> Type[BaseTransform]:
    """Get a transform class by name."""
    _auto_discover()
    if name not in _REGISTRY:
        raise ValueError(f"Transform '{name}' not found in registry.")
    return _REGISTRY[name]

def get_by_category(category: str) -> list[Type[BaseTransform]]:
    """Get all transforms in a specific category."""
    _auto_discover()
    return _CATEGORIES.get(category, [])
