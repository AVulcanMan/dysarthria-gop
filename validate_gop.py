"""Validate GoP scorers against severity labels at production / word / speaker level.

Reads a ``*_outputs.pkl`` produced by ``gop.py``. That pickle is a dict with:
    - "severity": list, one severity label per production (higher = WORSE)
    - "logits", "labels", "index_to_vocab": model-internals, not used here
    - "df": DataFrame with columns audio, phone, min, max, label, split
    - one key per scorer name (e.g. "NN-GoP") -> list (len = n productions) of
      per-phone numpy arrays. A production's word-level score is the mean of
      its per-phone array.

Convention: higher GoP score = better pronunciation; higher severity label =
worse. So we correlate each scorer's word-level score against the NEGATED
label -- a good scorer should then show a positive Kendall tau / Spearman rho.

Speaker/word resolution (best effort, in priority order):
    1. `df` already has `speaker` and/or `word` columns -> use directly.
    2. `--crosswalk CSV` maps anonymized audio basenames to speaker/word ->
       joined on the basename of the `audio` column.
    3. Neither available -> production-level statistics only.

CLI:
    python validate_gop.py --outputs_pkl test_dataset.csv_outputs.pkl \
        [--crosswalk crosswalk_anon_to_speaker.csv]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

_NON_SCORER_KEYS = {"severity", "logits", "labels", "index_to_vocab", "df"}


def _detect_scorer_keys(pkl_dict: dict) -> list:
    """Return keys that look like list-of-per-phone-arrays scorer outputs."""
    n = len(pkl_dict.get("severity", []))
    scorer_keys = []
    for key, value in pkl_dict.items():
        if key in _NON_SCORER_KEYS:
            continue
        if isinstance(value, list) and (n == 0 or len(value) == n):
            scorer_keys.append(key)
    return scorer_keys


def _word_level_scores(per_phone_arrays: list) -> np.ndarray:
    """Mean per-phone score -> one scalar score per production."""
    out = np.empty(len(per_phone_arrays))
    for i, arr in enumerate(per_phone_arrays):
        arr = np.asarray(arr, dtype=float)
        out[i] = np.nanmean(arr) if arr.size else np.nan
    return out


def _resolve_speaker_word(df: pd.DataFrame, crosswalk_df: pd.DataFrame = None):
    """Return (speaker_series, word_series), each aligned to df.index, or (None, None)."""
    speaker = df["speaker"] if "speaker" in df.columns else None
    word = df["word"] if "word" in df.columns else None

    if (speaker is not None) and (word is not None):
        return speaker, word

    if crosswalk_df is not None and "audio" in df.columns:
        cw = crosswalk_df.copy()
        # find the crosswalk's audio-like key column
        key_col = None
        for cand in ("audio", "anon_audio", "filename", "file"):
            if cand in cw.columns:
                key_col = cand
                break
        if key_col is None:
            key_col = cw.columns[0]
        cw["_basename"] = cw[key_col].astype(str).apply(lambda p: Path(p).name)
        basenames = df["audio"].astype(str).apply(lambda p: Path(p).name)
        merged = basenames.to_frame("_basename").merge(cw, on="_basename", how="left")
        merged.index = df.index
        spk = merged["speaker"] if "speaker" in merged.columns else speaker
        wrd = merged["word"] if "word" in merged.columns else word
        if spk is not None and wrd is not None:
            return spk, wrd

    return speaker, word


def _safe_corr(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan, np.nan
    tau = kendalltau(x, y).statistic
    rho = spearmanr(x, y).statistic
    return tau, rho


def run_validation(pkl_dict: dict, crosswalk_df: pd.DataFrame = None) -> pd.DataFrame:
    """Compute production/word/speaker-level Kendall tau & Spearman rho per scorer.

    Returns a tidy DataFrame with columns:
        scorer, level, n, kendall_tau, spearman_rho
    """
    severity = np.asarray(pkl_dict["severity"], dtype=float)
    neg_label = -severity
    df = pkl_dict.get("df")

    speaker = word = None
    if df is not None:
        speaker, word = _resolve_speaker_word(df, crosswalk_df)

    scorer_keys = _detect_scorer_keys(pkl_dict)
    rows = []

    for scorer in scorer_keys:
        word_scores = _word_level_scores(pkl_dict[scorer])

        tau, rho = _safe_corr(word_scores, neg_label)
        rows.append({"scorer": scorer, "level": "production", "n": len(word_scores),
                     "kendall_tau": tau, "spearman_rho": rho})

        if speaker is not None:
            spk_arr = np.asarray(speaker)
            frame = pd.DataFrame({"speaker": spk_arr, "score": word_scores, "neg_label": neg_label})
            grouped = frame.groupby("speaker", dropna=True).mean(numeric_only=True)
            tau_s, rho_s = _safe_corr(grouped["score"].values, grouped["neg_label"].values)
            rows.append({"scorer": scorer, "level": "speaker", "n": len(grouped),
                         "kendall_tau": tau_s, "spearman_rho": rho_s})

    return pd.DataFrame(rows, columns=["scorer", "level", "n", "kendall_tau", "spearman_rho"])


def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_pkl", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=None,
                         help="CSV mapping anonymized audio filename -> speaker,word")
    return parser.parse_args()


def main():
    args = _get_args()
    with open(args.outputs_pkl, "rb") as f:
        pkl_dict = pickle.load(f)

    crosswalk_df = None
    if args.crosswalk is not None:
        crosswalk_df = pd.read_csv(args.crosswalk)

    result = run_validation(pkl_dict, crosswalk_df)

    if result.empty:
        print("No scorer keys detected / no valid data.")
        return

    with pd.option_context("display.max_rows", None, "display.width", 120,
                            "display.float_format", "{:.4f}".format):
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
