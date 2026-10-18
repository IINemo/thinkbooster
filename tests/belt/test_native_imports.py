import importlib
import sys

import pytest

NAMES = ("luh", "train_luh")


def native_modules():
    return {
        name: module
        for name, module in sys.modules.copy().items()
        if name.split(".")[0] in NAMES
    }


@pytest.mark.parametrize("already_imported", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_loading_preserves_other_readers(
    tmp_path, monkeypatch, already_imported, fail
):
    from thinkbooster.belt.runtime import _native_reader_imports

    original = native_modules()
    for name in original:
        sys.modules.pop(name)
    native, external = tmp_path / "native", tmp_path / "external"
    for root, identity in ((native, "native"), (external, "external")):
        for package in NAMES:
            folder = root / package
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("from .child import identity\n")
            (folder / "child.py").write_text(f"identity = {identity!r}\n")
    monkeypatch.syspath_prepend(str(external))
    before_path = sys.path[:]
    try:
        before = {}
        if already_imported:
            before = {name: importlib.import_module(name) for name in NAMES}
        try:
            with _native_reader_imports(native):
                loaded = [importlib.import_module(name) for name in NAMES]
                assert [m.identity for m in loaded] == ["native", "native"]
                if fail:
                    raise RuntimeError("reader loading failed")
        except RuntimeError:
            if not fail:
                raise
        assert sys.path == before_path
        after = {name: importlib.import_module(name) for name in NAMES}
        assert [m.identity for m in after.values()] == ["external", "external"]
        if already_imported:
            assert all(after[name] is before[name] for name in NAMES)
        assert all(m.identity == "native" for m in loaded)
    finally:
        for name in native_modules():
            sys.modules.pop(name)
        sys.modules.update(original)
