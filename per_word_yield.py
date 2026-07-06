"""Per-word mild-band diagnostic yield ranking (PROJECT_CONTEXT.md sec 7, stimulus side #6).

For each word, restricted to the mild band (severity <= band), compute how well
GoP separates patient vs control speakers -- the "diagnostic yield" of that
word. Ranking words by yield tells you which stimuli are already doing the
diagnostic work, seeding the design of new/harder diagnostic phrases.

Per word:
    - if both classes have >= 2 productions: yield = ROC-AUC of -score as the
      patient-predictor (same orientation as mild_band_auc.py: AUC > 0.5 means
      low GoP flags patients).
    - otherwise (fewer than 2 of one class): fall back to Cohen's d between
      patient and control scores (sign-adjusted so more positive = more
      diagnostic in the same "low GoP -> patient" direction), reported as a
      distinguishable "yield_metric" of "cohens_d" instead of "auc".

Diagnosis ground truth: speaker-id prefix, "N*" = control else patient (see
mild_band_auc.py / PROJECT_CONTEXT.md sec 2).

CLI:
    python per_word_yield.py --outputs_pkl test_dataset.csv_outputs.pkl \
        [--crosswalk crosswalk_anon_to_speaker.csv] [--band 24.1] [--scorer NN-GoP]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from mild_band_auc import _diagnosis_from_speaker
from validate_gop import _detect_scorer_keys, _resolve_speaker_word, _word_level_scores


def _cohens_d(patient_scores: np.ndarray, control_scores: np.ndarray) -> float:
    """Cohen's d, oriented so positive = control scores higher than patient
    (i.e. same direction as 'low GoP flags patient')."""
    n1, n2 = len(control_scores), len(patient_scores)
    if n1 == 0 or n2 == 0:
        return np.nan
    v1, v2 = np.var(control_scores, ddof=1) if n1 > 1 else 0.0, np.var(patient_scores, ddof=1) if n2 > 1 else 0.0
    pooled_n = n1 + n2 - 2
    if pooled_n <= 0:
        pooled_sd = np.sqrt((v1 + v2) / 2) if (v1 + v2) > 0 else np.nan
    else:
        pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / pooled_n)
    if not pooled_sd or np.isnan(pooled_sd) or pooled_sd == 0:
        return np.nan
    return (np.mean(control_scores) - np.mean(patient_scores)) / pooled_sd


def per_word_yield(df_like: pd.DataFrame) -> pd.DataFrame:
    """Rank words by in-band patient-vs-control diagnostic yield.

    `df_like` must have columns: "word", "score", "diagnosis" (0=control,
    1=patient). Caller is responsible for pre-filtering to the mild band.

    Returns a DataFrame with columns: word, yield, yield_metric, n_control,
    n_patient, sorted by yield descending.
    """
    rows = []
    for word, g in df_like.groupby("word"):
        control_scores = g.loc[g["diagnosis"] == 0, "score"].to_numpy(dtype=float)
        patient_scores = g.loc[g["diagnosis"] == 1, "score"].to_numpy(dtype=float)
        n_control, n_patient = len(control_scores), len(patient_scores)

        if n_control >= 2 and n_patient >= 2:
            patient_predictor = -g["score"].to_numpy(dtype=float)
            auc = roc_auc_score(g["diagnosis"].to_numpy(dtype=int), patient_predictor)
            rows.append({"word": word, "yield": auc, "yield_metric": "auc",
                         "n_control": n_control, "n_patient": n_patient})
        else:
            d = _cohens_d(patient_scores, control_scores)
            rows.append({"word": word, "yield": d, "yield_metric": "cohens_d",
                         "n_control": n_control, "n_patient": n_patient})

    result = pd.DataFrame(rows, columns=["word", "yield", "yield_metric", "n_control", "n_patient"])
    result = result.sort_values("yield", ascending=False, na_position="last").reset_index(drop=True)
    return result


def _build_word_frame(pkl_dict: dict, scorer: str, band: float, crosswalk_df: pd.DataFrame = None) -> pd.DataFrame:
    df = pkl_dict.get("df")
    if df is None:
        raise SystemExit("No 'df' in pickle; cannot resolve words.")
    severity = np.asarray(pkl_dict["severity"], dtype=float)

    speaker, word = _resolve_speaker_word(df, crosswalk_df)
    if speaker is None or word is None:
        raise SystemExit("Could not resolve speaker/word (no df columns and no usable crosswalk).")

    scores = _word_level_scores(pkl_dict[scorer])
    speaker = np.asarray(speaker)
    word = np.asarray(word)
    diagnosis = np.array([_diagnosis_from_speaker(s) for s in speaker])

    frame = pd.DataFrame({"word": word, "score": scores, "severity": severity,
                          "speaker": speaker, "diagnosis": diagnosis})
    return frame[frame["severity"] <= band].reset_index(drop=True)


def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_pkl", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--band", type=float, default=24.1)
    parser.add_argument("--scorer", type=str, default="NN-GoP")
    return parser.parse_args()


def main():
    args = _get_args()
    with open(args.outputs_pkl, "rb") as f:
        pkl_dict = pickle.load(f)

    crosswalk_df = None
    if args.crosswalk is not None:
        crosswalk_df = pd.read_csv(args.crosswalk)

    scorer_keys = _detect_scorer_keys(pkl_dict)
    if args.scorer not in scorer_keys:
        raise SystemExit(f"Scorer '{args.scorer}' not found. Available: {scorer_keys}")

    band_frame = _build_word_frame(pkl_dict, args.scorer, args.band, crosswalk_df)
    result = per_word_yield(band_frame)

    with pd.option_context("display.max_rows", None, "display.width", 120,
                            "display.float_format", "{:.4f}".format):
        print(f"scorer={args.scorer} band<= {args.band}")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
