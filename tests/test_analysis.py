"""Tests for the new analysis scripts, using a hand-fabricated scores_acc dict
matching the schema documented in gop.py / validate_gop.py -- no real audio,
no model, no network.
"""
import pathlib
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from validate_gop import run_validation
from mild_band_auc import mild_band_auc
from per_word_yield import per_word_yield
import port_checkpoint


# ---------------------------------------------------------------------------
# validate_gop.run_validation
# ---------------------------------------------------------------------------

def _make_scores_acc():
    n = 8
    severity = [5, 5, 15, 15, 25, 25, 35, 35]  # higher = worse
    speaker = ["S1", "S1", "S2", "S2", "S3", "S3", "S4", "S4"]
    word = ["w1", "w2", "w1", "w2", "w1", "w2", "w1", "w2"]

    # Base scorer: perfectly anti-monotonic with severity (higher severity ->
    # lower score). Ties are kept aligned with the severity pairs (e.g. indices
    # 0,1 share both severity==5 and the same score) so Kendall tau-b/Spearman
    # rho are exactly 1 rather than penalized for "discordant" tie-breaking.
    base = [40 - s for s in severity]
    scaled = [0.5 * b for b in base]  # second scorer: same ranking, different scale

    df = pd.DataFrame({
        "audio": [f"prod{i}.wav" for i in range(n)],
        "phone": ["ah"] * n,
        "min": [0.0] * n,
        "max": [0.5] * n,
        "label": severity,
        "split": ["test"] * n,
        "speaker": speaker,
        "word": word,
    })

    scores_acc = {
        "severity": severity,
        "logits": [None] * n,
        "labels": [None] * n,
        "index_to_vocab": {0: "ah", 1: "(...)"},
        "df": df,
        "NN-GoP": [np.array([v]) for v in base],
        "MaxLogit-GoP": [np.array([v, v]) for v in scaled],  # 2-phone production
    }
    return scores_acc


def test_run_validation_returns_row_per_scorer_with_finite_correlations():
    scores_acc = _make_scores_acc()
    result = run_validation(scores_acc)

    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"scorer", "level", "n", "kendall_tau", "spearman_rho"}
    assert set(result["scorer"]) == {"NN-GoP", "MaxLogit-GoP"}
    # production-level and speaker-level rows for each of the 2 scorers.
    assert len(result) == 4
    assert set(result["level"]) == {"production", "speaker"}
    assert np.all(np.isfinite(result["kendall_tau"]))
    assert np.all(np.isfinite(result["spearman_rho"]))
    # Construction was perfectly anti-monotonic with severity -> tau/rho == 1.
    np.testing.assert_allclose(result["kendall_tau"].to_numpy(), 1.0, atol=1e-9)
    np.testing.assert_allclose(result["spearman_rho"].to_numpy(), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# mild_band_auc.mild_band_auc
# ---------------------------------------------------------------------------

def test_mild_band_auc_returns_dict_with_auc_and_p_in_unit_interval():
    severities = np.array([5, 5, 5, 5, 20, 20, 20, 20], dtype=float)
    speakers = np.array(["N1", "N1", "N2", "N2", "P1", "P1", "P2", "P2"])
    diagnoses = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Controls score high (good pronunciation), patients score low.
    scores = np.array([5.0, 5.2, 6.0, 5.8, -5.0, -4.8, -4.5, -5.2])

    rng = np.random.default_rng(0)
    result = mild_band_auc(scores, severities, speakers, diagnoses,
                            band=100.0, n_perm=200, rng=rng)

    assert isinstance(result, dict)
    assert "auc" in result and "p_value" in result
    assert 0.0 <= result["auc"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
    # Controls and patients are perfectly separated by construction -> AUC == 1.
    assert result["auc"] == pytest.approx(1.0)
    assert result["n_control"] == 4
    assert result["n_patient"] == 4


def test_mild_band_auc_empty_class_returns_nan_auc():
    severities = np.array([5.0, 5.0])
    speakers = np.array(["N1", "N1"])
    diagnoses = np.array([0, 0])  # no patients at all
    scores = np.array([1.0, 2.0])
    result = mild_band_auc(scores, severities, speakers, diagnoses, band=100.0)
    assert np.isnan(result["auc"])


# ---------------------------------------------------------------------------
# per_word_yield.per_word_yield
# ---------------------------------------------------------------------------

def test_per_word_yield_returns_ranked_dataframe():
    df_like = pd.DataFrame({
        "word": ["w1", "w1", "w1", "w1", "w2", "w2", "w2", "w2"],
        "score": [5.0, 5.2, -5.0, -4.8, 1.0, 1.1, 0.9, 1.05],
        "diagnosis": [0, 0, 1, 1, 0, 0, 1, 1],
    })
    result = per_word_yield(df_like)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["word", "yield", "yield_metric", "n_control", "n_patient"]
    assert set(result["word"]) == {"w1", "w2"}
    # Ranked descending by yield.
    assert (result["yield"].diff().dropna() <= 1e-9).all()
    # w1 has a much cleaner control/patient separation than w2 -> ranked first.
    assert result.iloc[0]["word"] == "w1"


# ---------------------------------------------------------------------------
# port_checkpoint: arguments.pkl PosixPath -> str shim only (no torch/best.pt).
# ---------------------------------------------------------------------------

class _FakeTrainArgs:
    """Stand-in for the argparse.Namespace pickled as arguments.pkl."""
    def __init__(self):
        self.model = "facebook/wav2vec2-xls-r-300m"
        self.use_conv_only = True
        self.reduce_vocab = False
        self.commonphone_csv = pathlib.PosixPath("/data/train/commonphone.csv.gz")


def test_port_arguments_pkl_converts_posixpath_attr_to_str(tmp_path):
    args_obj = _FakeTrainArgs()
    assert isinstance(args_obj.commonphone_csv, pathlib.PurePath)

    with open(tmp_path / "arguments.pkl", "wb") as f:
        pickle.dump(args_obj, f)

    returned_path = port_checkpoint.port_arguments_pkl(tmp_path)
    assert returned_path == tmp_path / "arguments.pkl"

    with open(tmp_path / "arguments.pkl", "rb") as f:
        reloaded = pickle.load(f)

    # The shim must coerce the PosixPath attribute to a plain str so it can be
    # unpickled/used on any platform.
    assert isinstance(reloaded.commonphone_csv, str)
    assert reloaded.commonphone_csv == "/data/train/commonphone.csv.gz"
    # Untouched, non-path attributes survive unchanged.
    assert reloaded.use_conv_only is True
    assert reloaded.model == "facebook/wav2vec2-xls-r-300m"


def test_port_checkpoint_skips_missing_best_pt(tmp_path):
    args_obj = _FakeTrainArgs()
    with open(tmp_path / "arguments.pkl", "wb") as f:
        pickle.dump(args_obj, f)

    result = port_checkpoint.port_checkpoint(tmp_path)
    assert result["arguments_pkl"] == str(tmp_path / "arguments.pkl")
    assert result["best_pt"] is None  # no best.pt in tmp_path -> cleanly skipped
