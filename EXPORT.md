# Export manifest

Take this file back to the data-bearing (patient-audio) environment. It summarizes
what changed in this session, what's new, and the exact commands to re-score a cohort.

## 1. Changed source files (bug fixes)

| File | Bug ID | What changed / why |
|---|---|---|
| `loss.py` | A1 | `_get_losses_lengths(logits, labels, log=True)` now feeds `F.nll_loss` **log-probabilities** (`log_softmax`) by default. `F.nll_loss` mathematically expects log-probs; feeding it plain `softmax` probabilities (still available via `log=False`, kept only for backward-compatible reproduction) silently computed the wrong loss. **This changes training** — a model retrained with the corrected loss is not numerically identical to one trained under the old code. |
| `gop.py` | A2 | `--temperature` default (7.8375) is now documented in `--help` as fit **only** for the original reference model/checkpoint; the CLI help explicitly says it must be re-fit per model via `temperature_scaling.py`. No silent reuse across checkpoints. |
| `gop.py` | A3 | `normalizer(scorer, prior)` now subtracts `np.log(prior + eps)` (log-domain) from logits before scoring, not `prior` itself. This is the mathematically correct prior-normalization (log-prior aligns with log-odds space of the logits); it changes the numeric value of **every** `Norm*-GoP` and `DNN-GoP` score vs. the old code. |
| `gop.py` | A4 | `entropy_gop_scorer` contract restored: it returns a **1-D per-phone `np.ndarray`** (one entropy value per phone in the production), matching every other scorer's shape contract — not a single pooled scalar. Convention: entropy is uncertainty, so **higher = worse** (opposite of the other scorers, where higher = better); downstream code doing comparative analysis across scorers should negate `Entropy-GoP`/`NormEntropy-GoP`/`ScaleEntropy-GoP` first. |
| `train_phone_recognizer.py` | A5 | Added `_str2bool` and switched `--use_conv_only` / `--reduce_vocab` (and `gop.py`'s `--ignore_nonexisting_vocab`) to use it instead of bare `type=bool`. `bool("False") == True` in Python, so e.g. `--use_conv_only False` was silently ignored before this fix and always evaluated truthy. |
| `train_phone_recognizer.py` | A6 | `_get_collator`'s inner `_collate` now derives each sample's true feature length from its **true unpadded raw waveform length** (recorded before the feature-extractor zero-pads the batch), instead of comparing padded audio samples against the `-100` label sentinel. The old logic conflated zero-valued waveform padding with the ignore-label convention, corrupting label alignment for batches with mixed-length audio. |
| `model.py` | A7 | `Wav2Vec2Recognizer._get_features` now calls `self.net.wav2vec2(inputs)[0]` (the `last_hidden_state`, 1024-d) instead of `self.net(inputs)[0]` (`projected_states`, 768-d). The old code returned a feature dimensionality that mismatched the classifier head built for this model. Note: this is the **full-transformer** path (`--use_conv_only False`); the local CPU pipeline only ever exercises `Wav2Vec2ConvRecognizer` (conv-only), which was already correct — this fix matters only if you ever run the full-transformer path locally. |

## 2. New files

| File | Purpose |
|---|---|
| `validate_gop.py` | Reads a `*_outputs.pkl` (from `gop.py`) and computes production-level and speaker-level Kendall tau / Spearman rho of each scorer's word-level (mean-per-production) score against negated severity. Auto-detects scorer keys in the pickle; resolves speaker/word from `df` columns or an optional `--crosswalk` CSV. |
| `mild_band_auc.py` | Restricts to the "mild band" (severity <= `--band`, default 24.1 — the zone where a human rater can't reliably separate patient from control) and computes patient-vs-control ROC-AUC using `-score` as the patient predictor, plus a speaker-level permutation p-value. Diagnosis ground truth comes from the speaker-id prefix (`N*` = control). |
| `per_word_yield.py` | Per word, within the mild band, ranks diagnostic yield (patient-vs-control AUC, or Cohen's d fallback when a class has fewer than 2 productions for that word) — used to identify which stimuli already carry diagnostic signal, to guide designing new/harder phrases. |
| `port_checkpoint.py` | Two independent portability shims for a training checkpoint directory: (a) re-pickles `arguments.pkl` with a `PosixPath` -> `PurePosixPath` shim and coerces any path-typed attributes to plain `str`, so it can be unpickled on a non-POSIX (e.g. Windows) machine; (b) reloads `best.pt` with `map_location="cpu"` and re-saves it, so CUDA-only tensors load on a CPU-only machine. `torch` is imported lazily so shim (a) alone works even without torch installed. |
| `tests/` | Synthetic-tensor pytest suite (no real audio, no model downloads, no network) covering the GoP scorer contracts, the loss.py A1 fix, and the new analysis scripts. See section 5. |

## 3. Exact commands to re-score your cohort

Run from the repo root, in order. Adjust paths to your environment.

```bash
# 1) Port a GPU/Linux-trained checkpoint dir so it loads here (CPU/Windows-safe).
#    Safe to run even if the checkpoint was already trained/ported locally (idempotent).
python port_checkpoint.py --model_dir path/to/exp/run_dir

# 2) Re-fit temperature scaling for THIS checkpoint (A2: do not reuse 7.8375 from
#    the reference model). Uses the model's dev/test split logits; prints the fit value.
python temperature_scaling.py \
    --dataset_csv path/to/test_dataset.csv.gz \
    --split dev \
    --model_path path/to/exp/run_dir

# 3) Run GoP scoring with the re-fit temperature. Produces
#    <dataset_csv_stem>_outputs.pkl with per-scorer per-production per-phone scores.
python gop.py \
    --dataset_csv path/to/test_dataset.csv.gz \
    --commonphone_csv path/to/commonphone.csv.gz \
    --model_path path/to/exp/run_dir \
    --temperature <value_from_step_2>

# 4) Validate: production/speaker-level Kendall tau & Spearman rho per scorer vs. severity.
python validate_gop.py \
    --outputs_pkl test_dataset.csv_outputs.pkl \
    [--crosswalk crosswalk_anon_to_speaker.csv]

# 5) Mild-band patient-vs-control AUC (the "hears what humans can't" test).
python mild_band_auc.py \
    --outputs_pkl test_dataset.csv_outputs.pkl \
    [--crosswalk crosswalk_anon_to_speaker.csv] \
    --scorer NN-GoP --band 24.1 --n_perm 5000

# 6) Per-word diagnostic-yield ranking, to seed new/harder stimulus design.
python per_word_yield.py \
    --outputs_pkl test_dataset.csv_outputs.pkl \
    [--crosswalk crosswalk_anon_to_speaker.csv] \
    --scorer NN-GoP --band 24.1
```

## 4. Behavior changes to be aware of

- **A1 (loss.py):** The corrected loss (`log=True`, log_softmax) changes what a freshly
  trained model converges to. If you need to exactly reproduce an old checkpoint's
  training run, pass `log=False` explicitly; otherwise re-train (or just re-score an
  already-trained checkpoint with `gop.py`, which does not use `loss.py` at inference time).
- **A2 (temperature):** `7.8375` is fit for the original reference model only. Every new
  checkpoint needs its own temperature via `temperature_scaling.py` (step 2 above) before
  trusting any `Scale*-GoP` score.
- **A3 (normalizer):** All `Norm*-GoP` and `DNN-GoP` scores are numerically different from
  any pickle produced before this fix. Don't compare old and new `*_outputs.pkl` files for
  those scorer columns; re-score from scratch.
- **A4 (Entropy-GoP):** Now a genuine per-phone array like every other scorer (previously
  a single scalar). Remember the sign convention: higher Entropy-GoP = more uncertain =
  **worse** pronunciation, the opposite of every other scorer in the pickle. Negate it
  before mixing with other scorers in comparative plots/tables.

## 5. Test suite

`tests/` is a synthetic-tensor pytest suite — no real audio, no model checkpoints, no
network calls.

- `tests/conftest.py` — stubs `librosa`/`praatio`/`praatio.textgrid`/`textgrids`/
  `tensorboard`/`torch.utils.tensorboard` in `sys.modules` (with real `ModuleSpec`s, so
  `transformers`' `importlib.util.find_spec` probing doesn't choke on them) so that
  `import gop` works in this environment even though `librosa`/`tensorboard` aren't
  installed. None of the stubbed functionality is exercised by the scorer functions
  under test.
- `tests/test_scorers.py` — synthetic `logits`/`labels` tensors; asserts every per-phone
  scorer returns a 1-D `np.ndarray` of length = number of non-ignored phones (explicit A4
  regression check on `entropy_gop_scorer`), a monotonicity check for `logit_gop_scorer`,
  `normalizer`-with-uniform-prior shift-invariance for `logit_margin_gop_scorer`, and
  `scaler`-with-`temperature=1` no-op checks.
- `tests/test_loss.py` — synthetic `logits`/`labels` with `-100`-ignored positions;
  asserts `log=True` losses are non-negative scalars and `log=False` losses go negative
  (A1 encoding), across all three loss functions.
- `tests/test_analysis.py` — hand-fabricated `scores_acc`-schema dict; asserts
  `run_validation` returns one row per scorer per level with finite tau/rho,
  `mild_band_auc` returns a dict with `auc`/`p_value` in `[0, 1]`, `per_word_yield`
  returns a yield-ranked `DataFrame`, and `port_checkpoint`'s `arguments.pkl`
  `PosixPath` -> `str` shim round-trips correctly (the `best.pt` torch-tensor shim is
  intentionally not exercised here).

Run with:

```bash
python -m pytest tests/ -q
```

Last run in this session: **32 passed, 2 warnings in 6.80s** (warnings are a pre-existing
NumPy 2.0 `__array_wrap__` deprecation notice from `gop.py`'s `normalizer`, not from the
test code).
