"""Shared test setup.

This repo's ``gop.py`` transitively imports ``dataset.py`` (needs ``librosa``),
``train_phone_recognizer.py`` (needs ``torch.utils.tensorboard`` -> the
``tensorboard`` pip package), and ``model.py`` (needs ``transformers``, which
*is* installed). None of ``librosa``/``praatio``/``textgrids``/``tensorboard``
are installed in this environment, and none of their actual functionality is
exercised by the pure-numeric GoP scorer functions under test, so we install
lightweight stub modules into ``sys.modules`` before anything imports ``gop``.

Each stub is given a real ``importlib.machinery.ModuleSpec`` (not just a bare
``types.ModuleType``) because ``transformers`` probes optional dependencies
via ``importlib.util.find_spec(name)``; if a name is already present in
``sys.modules`` with ``__spec__ is None``, ``find_spec`` raises
``ValueError`` instead of treating it as "not installed".
"""
import importlib.machinery
import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = mod
    return mod


def install_import_stubs() -> None:
    """Idempotently stub out the unavailable/unneeded heavy deps."""
    _stub_module("librosa")

    praatio = _stub_module("praatio")
    praatio_textgrid = _stub_module("praatio.textgrid")
    praatio.textgrid = praatio_textgrid

    _stub_module("textgrids")

    _stub_module("tensorboard")
    torch_utils_tb = _stub_module("torch.utils.tensorboard")
    if not hasattr(torch_utils_tb, "SummaryWriter"):
        torch_utils_tb.SummaryWriter = object


install_import_stubs()
