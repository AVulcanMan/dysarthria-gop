"""Synthetic-tensor tests for the per-phone GoP scorers in gop.py.

Importing gop.py transitively imports dataset.py (needs librosa, not
installed here). conftest.py installs stub modules for librosa/praatio/
textgrids/tensorboard before this file is collected, so `import gop` works
without any of those packages actually being present. If that stubbing
chain ever breaks in some other environment, we skip the whole module with
a clear reason rather than erroring out the run.
"""
import numpy as np
import pytest
import torch

gop = pytest.importorskip(
    "gop",
    reason="gop.py (and its dataset.py/model.py/train_phone_recognizer.py "
           "import chain) could not be imported even with the librosa/"
           "praatio/textgrids/tensorboard stubs installed by conftest.py",
)

V = 5  # vocab size
IGNORE_LABEL = 4


def _make_logits_labels(seed=0):
    """labels has consecutive runs per phone (incl. one ignored run) so that
    _phonewise_loop groups frames the same way gop.py's real usage does.

    Runs: phone 0 (3 frames), phone 1 (2 frames), IGNORE (2 frames),
    phone 2 (3 frames), phone 3 (2 frames) -> 4 non-ignore phones, T=12.
    """
    g = torch.Generator().manual_seed(seed)
    labels = torch.LongTensor(
        [0, 0, 0, 1, 1, IGNORE_LABEL, IGNORE_LABEL, 2, 2, 2, 3, 3]
    )
    logits = torch.randn(labels.shape[0], V, generator=g)
    return logits, labels


N_NON_IGNORE_PHONES = 4

ALL_SCORERS = [
    gop.gmm_gop_scorer,
    gop.nn_gop_scorer,
    gop.logit_margin_gop_scorer,
    gop.margin_gop_scorer,
    gop.logit_gop_scorer,
    gop.mean_prob_gop_scorer,
    gop.entropy_gop_scorer,
]


@pytest.mark.parametrize("scorer", ALL_SCORERS, ids=lambda f: f.__name__)
def test_scorer_returns_1d_array_of_correct_length(scorer):
    logits, labels = _make_logits_labels()
    out = scorer(logits, labels, IGNORE_LABEL)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert len(out) == N_NON_IGNORE_PHONES


def test_entropy_gop_scorer_is_an_array_not_a_scalar():
    # Explicit A4 regression check: entropy_gop_scorer must return one entropy
    # value PER non-ignored phone, not a single pooled scalar.
    logits, labels = _make_logits_labels()
    out = gop.entropy_gop_scorer(logits, labels, IGNORE_LABEL)
    assert isinstance(out, np.ndarray)
    assert out.shape == (N_NON_IGNORE_PHONES,)
    assert np.isscalar(out) is False


@pytest.mark.parametrize(
    "scorer", ALL_SCORERS, ids=lambda f: f.__name__
)
def test_scorer_values_are_finite(scorer):
    logits, labels = _make_logits_labels()
    out = scorer(logits, labels, IGNORE_LABEL)
    assert np.all(np.isfinite(out))


def test_logit_gop_scorer_monotonic_in_target_logit():
    # Two phones, each with 3 frames, same vocab size. Phone A's target-class
    # logit is raised well above phone B's -> logit_gop_scorer(A) > logit_gop_scorer(B).
    labels = torch.LongTensor([0, 0, 0, 1, 1, 1])
    logits = torch.zeros(6, V)
    # Give every frame a mildly-random baseline so this isn't a degenerate all-zeros case.
    g = torch.Generator().manual_seed(1)
    logits += 0.01 * torch.randn(6, V, generator=g)

    logits[0:3, 0] = 5.0   # phone 0's target logit: high
    logits[3:6, 1] = -5.0  # phone 1's target logit: low

    ignore_label = 99  # not present in labels -> nothing gets ignored
    scores = gop.logit_gop_scorer(logits, labels, ignore_label)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_normalizer_uniform_prior_shifts_logits_by_constant():
    logits, labels = _make_logits_labels()
    uniform_prior = np.ones(V) / V

    plain = gop.logit_margin_gop_scorer(logits, labels, IGNORE_LABEL)
    normalized = gop.normalizer(gop.logit_margin_gop_scorer, uniform_prior)(
        logits, labels, IGNORE_LABEL
    )

    # A uniform prior means log(prior) is the SAME constant for every vocab
    # entry, i.e. normalizer subtracts a constant from every logit. LogitMargin
    # is a difference between two logits (target vs. best-of-the-rest), so it's
    # shift-invariant: the constant cancels and the score must be unchanged.
    np.testing.assert_allclose(normalized, plain, rtol=1e-5, atol=1e-6)


def test_normalizer_nonuniform_prior_changes_non_shift_invariant_scorer():
    logits, labels = _make_logits_labels()
    # A skewed (non-uniform) prior is NOT a constant shift across vocab
    # entries, so a scorer that isn't shift-invariant (plain MaxLogit) should
    # change under normalization.
    skewed_prior = np.array([0.9, 0.025, 0.025, 0.025, 0.025])

    plain = gop.logit_gop_scorer(logits, labels, IGNORE_LABEL)
    normalized = gop.normalizer(gop.logit_gop_scorer, skewed_prior)(
        logits, labels, IGNORE_LABEL
    )
    assert not np.allclose(normalized, plain)


def test_scaler_temperature_one_is_a_noop():
    logits, labels = _make_logits_labels()
    for scorer in (gop.logit_gop_scorer, gop.logit_margin_gop_scorer, gop.entropy_gop_scorer):
        plain = scorer(logits, labels, IGNORE_LABEL)
        scaled = gop.scaler(scorer, 1.0)(logits, labels, IGNORE_LABEL)
        np.testing.assert_allclose(scaled, plain, rtol=1e-5, atol=1e-6)


def test_scaler_temperature_other_than_one_changes_temperature_sensitive_scorer():
    logits, labels = _make_logits_labels()
    plain = gop.logit_gop_scorer(logits, labels, IGNORE_LABEL)
    scaled = gop.scaler(gop.logit_gop_scorer, 2.0)(logits, labels, IGNORE_LABEL)
    np.testing.assert_allclose(scaled, plain / 2.0, rtol=1e-5, atol=1e-6)
