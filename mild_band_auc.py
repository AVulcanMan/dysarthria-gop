"""Mild-band patient-vs-control AUC test (PROJECT_CONTEXT.md sec 6b).

Within the "mild band" (perceptual severity <= band, i.e. at/below the control
ceiling -- the zone where a human rater can't reliably tell patient from
control), does GoP still separate patient from control speakers? Diagnosis
ground truth is read from the SPEAKER-ID PREFIX: an id starting with "N" is a
control, anything else is a patient (see PROJECT_CONTEXT.md sec 2 table).

Because higher GoP = better and we want "patient" to be the positive class,
the AUC is computed with `-score` as the patient-predictor: AUC > 0.5 means
low GoP flags patients (correctly), including patients the rater scored as
normal (that's the "hears what humans can't" signal).

Significance is assessed with a SPEAKER-level permutation test: shuffle the
diagnosis label across speakers (not productions) n_perm times and see how
often the permuted AUC is >= the observed one.

CLI:
    python mild_band_auc.py --outputs_pkl test_dataset.csv_outputs.pkl \
        [--crosswalk crosswalk_anon_to_speaker.csv] [--band 24.1] \
        [--scorer NN-GoP] [--n_perm 5000]
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from validate_gop import _detect_scorer_keys, _resolve_speaker_word, _word_level_scores


def _diagnosis_from_speaker(speaker: str) -> int:
    """1 = patient, 0 = control. Control iff the speaker id starts with 'N'."""
    return 0 if str(speaker).upper().startswith("N") else 1


def mild_band_auc(scores: np.ndarray, severities: np.ndarray, speakers: np.ndarray,
                   diagnoses: np.ndarray, band: float = 24.1, n_perm: int = 5000,
                   rng: np.random.Generator = None) -> dict:
    """Patient-vs-control AUC of GoP within the mild (low-severity) band.

    Parameters mirror same-length per-production arrays. `diagnoses` is 0/1
    (0 = control, 1 = patient) per production.

    Returns a dict with auc, p_value, n_control, n_patient, n_productions,
    n_speakers_control, n_speakers_patient, band.
    """
    scores = np.asarray(scores, dtype=float)
    severities = np.asarray(severities, dtype=float)
    speakers = np.asarray(speakers)
    diagnoses = np.asarray(diagnoses, dtype=int)

    in_band = severities <= band
    scores_b = scores[in_band]
    diag_b = diagnoses[in_band]
    speakers_b = speakers[in_band]

    result = {
        "band": band,
        "n_productions": int(in_band.sum()),
        "n_control": int((diag_b == 0).sum()),
        "n_patient": int((diag_b == 1).sum()),
        "n_speakers_control": int(len(set(speakers_b[diag_b == 0]))),
        "n_speakers_patient": int(len(set(speakers_b[diag_b == 1]))),
        "auc": np.nan,
        "p_value": np.nan,
    }

    if result["n_control"] == 0 or result["n_patient"] == 0:
        return result

    patient_predictor = -scores_b
    observed_auc = roc_auc_score(diag_b, patient_predictor)
    result["auc"] = observed_auc

    # speaker-level permutation: shuffle diagnosis label across unique speakers
    uniq_speakers = np.array(sorted(set(speakers_b)))
    speaker_diag = {}
    for spk in uniq_speakers:
        d = diag_b[speakers_b == spk]
        speaker_diag[spk] = int(round(d.mean()))  # should be constant per speaker
    diag_values = np.array([speaker_diag[s] for s in uniq_speakers])

    if rng is None:
        rng = np.random.default_rng(0)

    if len(set(diag_values.tolist())) < 2:
        result["p_value"] = np.nan
        return result

    ge_count = 0
    for _ in range(n_perm):
        perm_diag_values = rng.permutation(diag_values)
        perm_map = dict(zip(uniq_speakers, perm_diag_values))
        perm_diag_b = np.array([perm_map[s] for s in speakers_b])
        if len(set(perm_diag_b.tolist())) < 2:
            continue
        perm_auc = roc_auc_score(perm_diag_b, patient_predictor)
        if perm_auc >= observed_auc:
            ge_count += 1

    result["p_value"] = ge_count / n_perm
    return result


def _get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs_pkl", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, default=None)
    parser.add_argument("--band", type=float, default=24.1)
    parser.add_argument("--scorer", type=str, default="NN-GoP")
    parser.add_argument("--n_perm", type=int, default=5000)
    return parser.parse_args()


def main():
    args = _get_args()
    with open(args.outputs_pkl, "rb") as f:
        pkl_dict = pickle.load(f)

    crosswalk_df = None
    if args.crosswalk is not None:
        crosswalk_df = pd.read_csv(args.crosswalk)

    df = pkl_dict.get("df")
    severity = np.asarray(pkl_dict["severity"], dtype=float)

    scorer_keys = _detect_scorer_keys(pkl_dict)
    if args.scorer not in scorer_keys:
        raise SystemExit(f"Scorer '{args.scorer}' not found. Available: {scorer_keys}")

    scores = _word_level_scores(pkl_dict[args.scorer])

    if df is None:
        raise SystemExit("No 'df' in pickle; cannot resolve speakers for diagnosis.")
    speaker, _word = _resolve_speaker_word(df, crosswalk_df)
    if speaker is None:
        raise SystemExit("Could not resolve speaker ids (no df.speaker and no usable crosswalk).")

    speaker = np.asarray(speaker)
    diagnoses = np.array([_diagnosis_from_speaker(s) for s in speaker])

    rng = np.random.default_rng(0)
    result = mild_band_auc(scores, severity, speaker, diagnoses,
                            band=args.band, n_perm=args.n_perm, rng=rng)

    print(f"scorer={args.scorer} band<= {args.band}")
    print(f"n_productions={result['n_productions']} "
          f"(control={result['n_control']}, patient={result['n_patient']})")
    print(f"n_speakers: control={result['n_speakers_control']}, patient={result['n_speakers_patient']}")
    print(f"AUC (patient vs control, -score as patient-predictor) = {result['auc']:.4f}")
    print(f"speaker-permutation p-value (n_perm={args.n_perm}) = {result['p_value']:.4f}")


if __name__ == "__main__":
    main()
