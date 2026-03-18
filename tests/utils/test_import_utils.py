#!/usr/bin/env python

import importlib.metadata
from types import SimpleNamespace

from lerobot.utils.import_utils import is_package_available


def test_is_package_available_accepts_importable_module_without_metadata(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object() if name == "transformers" else None)

    def fake_version(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", fake_version)
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: SimpleNamespace(__version__="4.99.0-local") if name == "transformers" else None,
    )

    available, version = is_package_available("transformers", return_version=True)

    assert available is True
    assert version == "4.99.0-local"
