"""Synthetic-tensor tests for loss.py, encoding the A1 fix.

A1: F.nll_loss expects LOG-probabilities. `_get_losses_lengths(..., log=True)`
(the corrected default) feeds it log_softmax, so every per-position loss is
`-log_prob >= 0` and the batch-mean must be a non-negative scalar. The old
(buggy) behavior fed it plain softmax probabilities instead
(`log=False`, kept only for backward-compatible reproduction); nll_loss then
computes `-prob` per position, which is negative for any prob in (0, 1), so
that path can -- and with generic random logits, reliably does -- go negative.
"""
import torch

from loss import phonewise_average_loss, samplewise_average_loss, ctc_like_loss

torch.manual_seed(0)

B, T, V = 3, 6, 5


def _make_logits_labels():
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    # Ignore a few trailing positions in each sample, like padded frames.
    labels[0, 4:] = -100
    labels[1, 5:] = -100
    return logits, labels


def test_phonewise_average_loss_log_true_is_nonnegative_scalar():
    logits, labels = _make_logits_labels()
    loss = phonewise_average_loss(logits, labels, log=True)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_phonewise_average_loss_log_false_can_go_negative():
    logits, labels = _make_logits_labels()
    loss = phonewise_average_loss(logits, labels, log=False)
    assert loss.dim() == 0
    # Every kept position contributes -softmax_prob(target), which is strictly
    # negative (probabilities lie in (0, 1)), so the weighted sum must be < 0.
    assert loss.item() < 0.0


def test_samplewise_average_loss_log_true_is_nonnegative_scalar():
    logits, labels = _make_logits_labels()
    loss = samplewise_average_loss(logits, labels, log=True)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_samplewise_average_loss_log_false_can_go_negative():
    logits, labels = _make_logits_labels()
    loss = samplewise_average_loss(logits, labels, log=False)
    assert loss.item() < 0.0


def test_ctc_like_loss_log_true_is_nonnegative_scalar():
    logits, labels = _make_logits_labels()
    loss = ctc_like_loss(logits, labels, log=True)
    assert loss.dim() == 0
    assert loss.item() >= 0.0


def test_ctc_like_loss_log_false_can_go_negative():
    logits, labels = _make_logits_labels()
    loss = ctc_like_loss(logits, labels, log=False)
    assert loss.item() < 0.0
