"""Port a GPU/Linux training checkpoint dir so it loads on CPU/Windows.

Two independent portability shims (see PROJECT_CONTEXT.md sec 8):

  (a) `arguments.pkl` may contain `pathlib.PosixPath` objects, which raise
      `NotImplementedError: cannot instantiate PosixPath` when unpickled on a
      non-POSIX platform. Fixed by unpickling with a custom `Unpickler` that
      remaps `pathlib.PosixPath` -> `pathlib.PurePosixPath` (a pure,
      platform-independent path type that can always be instantiated), then
      coercing any path-typed attributes on the resulting object to plain
      `str` and re-pickling in place.

  (b) `best.pt` may hold CUDA tensors, which raise `RuntimeError: Attempting
      to deserialize object on a CUDA device` when loaded without a GPU.
      Fixed with `torch.save(torch.load(path, map_location="cpu"), path)`.

torch is imported LAZILY (only inside the function that needs it) so this
module can be imported / the arguments.pkl shim can run in environments
without torch installed. No model weights are downloaded.

CLI:
    python port_checkpoint.py --model_dir path/to/exp/run_dir
"""

import argparse
import pathlib
import pickle
from pathlib import Path


class _PosixToPureUnpickler(pickle.Unpickler):
    """Unpickler that maps pathlib.PosixPath -> pathlib.PurePosixPath.

    PosixPath can't be instantiated on Windows (or vice versa for WindowsPath
    on POSIX); PurePosixPath is a pure path type with no OS dependency, so it
    can always be constructed regardless of the host platform.
    """

    def find_class(self, module, name):
        if module == "pathlib" and name == "PosixPath":
            return pathlib.PurePosixPath
        if module == "pathlib" and name == "WindowsPath":
            return pathlib.PureWindowsPath
        return super().find_class(module, name)


def _coerce_path_attrs_to_str(obj):
    """Walk obj.__dict__ (if any) and stringify any path-like attribute values."""
    d = getattr(obj, "__dict__", None)
    if d is None:
        return obj
    for key, value in list(d.items()):
        if isinstance(value, (pathlib.PurePath,)):
            d[key] = str(value)
    return obj


def port_arguments_pkl(model_dir: Path) -> Path:
    """Re-pickle `model_dir/arguments.pkl` with PosixPath shimmed + path attrs stringified.

    Returns the path that was rewritten.
    """
    args_path = Path(model_dir) / "arguments.pkl"
    with open(args_path, "rb") as f:
        args = _PosixToPureUnpickler(f).load()

    args = _coerce_path_attrs_to_str(args)

    with open(args_path, "wb") as f:
        pickle.dump(args, f)

    return args_path


def port_best_pt(model_dir: Path) -> Path:
    """Reload `model_dir/best.pt` mapped to CPU and re-save it in place.

    Imports torch lazily; raises a clear error if torch is not installed.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "port_best_pt requires torch, which is not installed in this "
            "environment. Install torch (CPU build is fine) to port best.pt."
        ) from e

    best_pt_path = Path(model_dir) / "best.pt"
    state = torch.load(best_pt_path, map_location="cpu")
    torch.save(state, best_pt_path)
    return best_pt_path


def port_checkpoint(model_dir: Path) -> dict:
    """Run both shims against a checkpoint directory. Returns paths touched."""
    model_dir = Path(model_dir)
    result = {}

    args_path = model_dir / "arguments.pkl"
    if args_path.exists():
        result["arguments_pkl"] = str(port_arguments_pkl(model_dir))
    else:
        result["arguments_pkl"] = None

    best_pt_path = model_dir / "best.pt"
    if best_pt_path.exists():
        result["best_pt"] = str(port_best_pt(model_dir))
    else:
        result["best_pt"] = None

    return result


def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True,
                         help="Directory containing arguments.pkl and/or best.pt")
    return parser.parse_args()


def main():
    args = _get_args()
    result = port_checkpoint(args.model_dir)

    if result["arguments_pkl"]:
        print(f"Re-pickled: {result['arguments_pkl']}")
    else:
        print(f"No arguments.pkl found in {args.model_dir}; skipped.")

    if result["best_pt"]:
        print(f"Re-saved (CPU): {result['best_pt']}")
    else:
        print(f"No best.pt found in {args.model_dir}; skipped.")


if __name__ == "__main__":
    main()
