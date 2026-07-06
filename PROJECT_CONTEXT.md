# Project Context — Early-Dysarthria Detection via Goodness-of-Pronunciation

_Last updated: 2026-07-02_

A living handoff doc: the goal, the data, the pipeline, what works, what's broken, and
the open research directions. Read `models/README.md` and the two memory notes
(`gop-yeo-project`, `yeo-repo-fulltransformer-patches`) alongside this.

---

## 1. Goal (the "why")

Build a speech model that **flags degraded pronunciation earlier and more sensitively
than a human expert can**, for early detection of neurodegenerative disease
(ALS, PD, HD). Two coupled ambitions:

1. **Model side** — a scorer that "hears" sub-perceptual abnormality: the region where
   the model's signal rises but a trained clinician still rates the speech as normal
   (see `hypothesis.png` — the gap between the green/model curve and the flat part of
   the red/expert curve).
2. **Stimulus side** — design **more challenging diagnostic phrases** that stress the
   speech-motor system so mild/early patients fail measurably on them, moving the
   discriminating part of the severity curve leftward (earlier).

**Key reframing learned this session:** Goodness-of-Pronunciation (GoP) is anchored to
perceptual salience, so it is roughly *capped at human level*. To beat the human you must
(a) supervise on a label that is **not** human perception (diagnosis / progression), and
(b) measure acoustic dimensions the ear discards. See §7.

---

## 2. Data

**Validation cohort: 12 speakers × 20 words = 240 productions.** Isolated multisyllabic
words of graded phonetic complexity. Located under `gop_yeo/test_corpus/<SPEAKER>/`.

| Group | Speakers | Notes |
|---|---|---|
| Control | N495, N514, NF100 | `N*` prefix = neurotypical control |
| ALS | A4017, A4030, A4040 | `A4*` prefix |
| PD | PD08, PD09, PD17 | |
| HD | HDF02, HDM01 | Huntington's |
| (other) | TSGM22 | one extra speaker in the test set |

- **Label = articulatory-precision / severity VAS**, per-production, ~1–99, averaged
  over 5 raters. In `Precision Ratings.Complexity Score_All Data.xlsx` (sheet
  "Compiled Data"). Also `gop_yeo/test_dataset.csv.gz` column `label`.
- **⚠ LABEL ORIENTATION (settled):** higher = **WORSE** (imprecision/severity), NOT
  better. Verified 3 ways (population means Control 11.9 < ALS 29.4 < HD 33.8 < PD 42.6;
  independent clinician severity ratings agree; crosswalk join is 240/240 correct).
  So **GoP correlating *negatively* with the raw label is CORRECT** (high GoP = better).
  `validate_gop.py` negates the label internally so all scorers report positive.
- **No per-phone expert labels** → validate at word / speaker level only.
- **Diagnosis ground truth** is intrinsic in the speaker-id prefix (see table) — this is
  the independent-of-perception label that makes the "beats-the-expert" test possible (§6).
- **Ratings workbooks** in `ratings/`: `ALS Perceptual Ratings+Recording Quality.xlsx`,
  `ALS Severity Ratings_All Participants.xlsx`, `PD Perceptual Ratings+Recording Quality.xlsx`.
- **Word sharing:** 19 of 21 words are spoken by all 12 speakers → each word yields a
  12-point curve spanning the full severity range (useful for per-word analysis).
- **Crosswalk:** `gop_yeo/crosswalk_anon_to_speaker.csv` maps anon `NNN.wav` →
  speaker + word (recovered by content-hash match). Broader `complexity II 2019/`
  collection = 97 speakers (ALS/PD/Control, no HD) but mostly unrated — future expansion.

---

## 3. Method — the GoP framework (Yeo et al.)

- **Acoustic model:** `facebook/wav2vec2-xls-r-300m`, **conv-only** path (frozen 512-d
  conv features → linear phone head), `reduce_vocab=True` → 39 reduced-IPA phones.
- **GoP scorers:** 15 UQ-GoP variants computed in `gop.py` (GMM-, NN-, DNN-, Entropy-,
  Margin-, MaxLogit-, and Norm*/Scale* variants). Each produces a **per-phone** score;
  word-level = mean over phones.
- **Vendored repo:** `gop_yeo/dysarthria-gop/` (Yeo clone; runs conv-only on CPU).

---

## 4. Current pipeline (end to end)

```
raw wavs (test_corpus/<spk>/<spk>_<Word>1.wav, 16 kHz, "Say X again" carrier)
   │
   ├─ MFA alignment (conda `aligner` env; english_mfa IPA acoustic + custom dict)
   │     → phone segments (min/max times)
   │
   ├─ test_dataset.csv.gz   (audio, phone, min, max, label, split)  ← model input
   │
   ├─ gop.py  --dataset_csv test_dataset.csv.gz
   │          --commonphone_csv arctic_strong.csv.gz
   │          --model_path ../models/{strong|weak}
   │     → dysarthria-gop/test_dataset.csv_outputs.pkl   (per-production, 15 scorers)
   │       (weak baseline saved as test_dataset.csv_outputs.WEAK.pkl)
   │
   └─ validate_gop.py
         → word- & speaker-level Kendall τ / Spearman ρ per scorer (label negated)
```

**Recognizer training** (produces the phone head behind GoP):
```
prepare_arctic[_strong].py            # build ARCTIC training CSV
train_phone_recognizer.py            # conv-only XLS-R → linear head, phonewise_average loss
   → dysarthria-gop/exp/<run>/{best.pt, index_to_vocab.pkl, arguments.pkl}
```
Register a run as the scorer by copying those 3 files into `models/weak/` or `models/strong/`.

---

## 5. The two recognizers (weak vs strong)

| | **weak** | **strong** |
|---|---|---|
| Corpus | `arctic.csv.gz`, split by speaker | `arctic_strong.csv.gz`, split by utterance-id |
| Speakers in train | 2 (bdl, rms — both male) | 4 (bdl, rms, clb, slt — gender-balanced) |
| Train utts | 700 | 1200 (~1.7×) |
| Epochs | 10 | 20 |
| ARCTIC test frame acc | 0.526 | (higher; not the point — see below) |
| Source exp dir | `arctic_convhead_full_2026-07-01T10-32-36` | `arctic_conv_gpu_2026-07-02T11-35-03` (GPU) |

The current `models/strong/` holds the **GPU conv run** `arctic_conv_gpu_...T11-35-03`
(there is also a separate `arctic_strong_2026-07-02T10-47-09` CPU run — NOT the one
registered; swap if intended).

---

## 6. Results so far (the honest picture)

### 6a. Global severity correlation — strong model REGRESSED
Registering the GPU-strong model and running `gop.py` + `validate_gop.py`:

- Weak baseline (README): word τ ≈ 0.14–0.19; **speaker Spearman up to ≈ 0.70**
  (Entropy-/GMM-/NormEntropy-GoP).
- Strong (this run): every scorer collapsed toward 0. Best word τ = 0.116 (DNN-GoP);
  headline NormLogit speaker ρ only 0.224; some scorers went negative.

**A stronger clean-ARCTIC recognizer did NOT improve — it hurt — patient severity
correlation.** Likely cause: sharper recognizer → overconfident logits on
out-of-distribution dysarthric speech → flatter GoP-vs-severity signal. Root bottleneck
(from README): on patient speech, frame accuracy ≈ 0.315 and **barely varies with
severity**. Clean-speech recognizer quality is the wrong lever.

### 6b. The mild-band gap test — the promising thread
Global correlation is the WRONG metric for the goal (it rewards tracking the full
severity range, and ties at the floor/ceiling of a true expert-sigmoid actually *lower*
Kendall τ). The right test: **within the low-severity overlap band (perceptual severity
≤ 24.1, the control ceiling), does GoP separate patient from control — where the rater
can't?** (Independent ground truth = diagnosis from id prefix.)

| | STRONG | WEAK |
|---|---|---|
| best GoP AUC (patient vs control, in band) | **NN-GoP 0.655** | 0.533 |
| speaker-permutation p | **0.026** | 0.120 (ns) |
| perceptual-rating baseline AUC in band | 0.709 | 0.709 |

Reading:
- **Real independent early signal exists** (0.655, p=0.026, not driven by one speaker).
- **The "strong" model is vindicated here** — it carries the early signal; the weak model
  is at chance. Global τ was misleading us.
- **But "beats the expert" is NOT yet proven** — the perceptual rating still separates
  better (0.709) in this band, because the band isn't a clean "expert-is-blind" zone.
  Decisive follow-up = severity-matched / residual-GoP AUC, or a tighter floor band.
- **Caveat:** only 3 control speakers → directional, not clinical.

### 6c. What the hypothesis curve actually looks like (`hypothesis_real.png`)
Plotting real GoP (strong, NN-GoP, normalized penalty) vs severity:
- **Single random word ("Violate", 12 spk): essentially flat noise**, Spearman 0.03,
  sometimes reversed (controls high, severe patients mid). Per-word GoP is dominated by
  word phonetics + voice + channel, not disease.
- **All words per-speaker avg:** still flat / slightly wrong direction (ρ = −0.22).
- The clean sigmoid/step/linear curves in `hypothesis.png` are **aspirational**; the
  real signal only emerges after heavy aggregation (the §6b AUC), not as a usable
  single-word curve. → need SNR-raising: word-difficulty normalization, calibration head.

---

## 7. Open research directions (how to actually reach the goal)

### Model side — "hear what humans can't"
1. **Supervise on diagnosis / progression, not perceived severity** (biggest lever).
   GoP anchored to a perceptual score is ceilinged at human level. Train on
   patient-vs-control, ideally rate-of-change / longitudinal (earliest signal is change
   vs a person's own baseline).
2. **Anomaly-detection framing:** densely model the *healthy* distribution (many
   controls, word-normalized), score deviation / low likelihood. Statistically abnormal
   yet perceptually normal = literally "hears what humans can't." CPU-feasible on
   existing XLS-R features. **Recommended next prototype** — compare its mild-band AUC
   to the 0.655 GoP baseline; decides "keep improving GoP" vs "switch paradigm."
3. **Add features the ear discards:** voice-source (jitter, shimmer, HNR, microtremor),
   articulatory kinematics (VOT, formant slopes, spectral moments, coarticulation),
   and **cross-repetition variability**. Fuse with GoP (orthogonal information).
4. **Kill nuisance variance first:** z-score each production against that word's healthy
   distribution; per-speaker baseline; aggregate many productions. (This is what turns
   the §6c noise cloud into signal.)
5. **⚠ Confound guard (dominant failure mode):** models "hear" microphone / room /
   cohort / age / sex, not disease. Use matched controls, channel normalization,
   hold-out by recording site. Distrust any large AUC until confounds are ruled out.

### Stimulus side — challenging diagnostic phrases
Sensitivity = model × stimulus. Design phrases that exceed a mild system's compensatory
reserve:
1. **Rapid place-of-articulation switching** (labial↔coronal↔velar; DDK "pa-ta-ka").
2. **Cross-place consonant clusters** ("sixths", "strengths").
3. **Long low-frequency multisyllables** (already: "Preposterous", "Ventriloquist").
4. **Length + rate stress**; **repetition** to expose inconsistency.
5. **Disease-specific targeting** (also aids differential dx): ALS → lingual DDK +
   nasality; PD → long prosodic-range phrases (decay/festination); HD → timing-variability.
6. **Closed-loop / data-driven design:** rank each word by mild-band diagnostic yield
   (§6b sliced per word), find which phonetic features drive it, compose new phrases
   dense in those features, iterate. **Recommended next analysis** — per-word diagnostic
   yield vs phonetic complexity; validates "harder = more diagnostic" and gives the
   recipe for the next phrase set.

---

## 8. Known bugs / gotchas / environment

- **Environment: CPU-only** (i7-13700, 32 GB, Intel UHD, no CUDA). Patient audio stays
  on this machine (clinical). Windows; PowerShell primary + Bash (git-bash) available.
  Python 3.12 venv at `gop_yeo/venv`. Load audio via `soundfile`, not `torchaudio.load`.
- **GPU-artifact portability (hit this session):** checkpoints trained on the GPU/Linux
  box need two fixes before they load on this CPU/Windows machine:
  1. `arguments.pkl` contains `PosixPath` objects → `NotImplementedError: cannot
     instantiate PosixPath`. Fix: unpickle with a `PosixPath→PurePosixPath` shim,
     coerce path attrs to `str`, re-pickle. (gop.py only needs `use_conv_only`,
     `model`, `reduce_vocab`.)
  2. `best.pt` holds CUDA tensors → `RuntimeError: Attempting to deserialize on a CUDA
     device`. Fix: `torch.save(torch.load(p, map_location="cpu"), p)`.
  Both already applied to the registered `models/strong/`.
- **Vendored full-transformer path is broken** (`--use_conv_only False`): never
  exercised upstream. 3 patches exist ONLY in `gop_yeo/argon_gpu/code/` (GPU export
  kit), NOT the local clone: (1) argparse `type=bool` → real `_str2bool` (bool("False")
  is True, so the flag was silently ignored); (2) `Wav2Vec2Recognizer._get_features`
  `self.net(inputs)[0]` (768-d projected_states) → `self.net.wav2vec2(inputs)[0]`
  (1024-d last_hidden_state); (3) optional `textgrids`/`praatio` import. Local CPU path
  only ever runs conv-only, which is correct as-is.
- **MFA OOV risk:** new words must be OOV-checked before aligning ("procrastinate" once
  collapsed to `spn` for all 12 speakers). `custom_english_mfa.dict` adds missing words
  via g2p. Any dataset expansion → re-check OOV first.
- **Metric mismatch:** global Kendall/Spearman vs severity is the WRONG objective for
  early detection (see §6b/§6c). Use mild-band / residual AUC against diagnosis.
- **Small n:** 3 control speakers → all diagnosis results are directional.

---

## 9. Key files

| Path | What |
|---|---|
| `gop_yeo/test_dataset.csv.gz` | model input (audio, phone, min/max, label, split) |
| `gop_yeo/dysarthria-gop/gop.py` | computes 15 GoP scorers → `*_outputs.pkl` |
| `gop_yeo/dysarthria-gop/test_dataset.csv_outputs.pkl` | strong per-production scores |
| `gop_yeo/dysarthria-gop/test_dataset.csv_outputs.WEAK.pkl` | weak baseline scores |
| `gop_yeo/validate_gop.py` | word/speaker correlations (label negated) |
| `gop_yeo/models/{weak,strong}/` | registered recognizer checkpoints |
| `gop_yeo/models/README.md` | weak vs strong spec + STEP 7 commands |
| `gop_yeo/crosswalk_anon_to_speaker.csv` | anon wav → speaker + word |
| `gop_yeo/argon_gpu/code/` | GPU export kit (patched full-transformer path) |
| `ratings/*.xlsx` | perceptual + severity ratings |
| `hypothesis.png` | conceptual goal curve (expert sigmoid / ASR linear / casual step) |
| `hypothesis_real.png` | real GoP vs those ideals (shows single-word signal is noise) |

---

## 10. Immediate next steps (recommended)

1. **Decisive gap test:** residual / severity-matched patient-vs-control AUC (does GoP
   beat the rating where the rating is at chance?). Confirms or closes "hears what humans
   can't."
2. **Anomaly-detection prototype:** one-class model on control XLS-R features
   (word-normalized) → mild-band AUC vs the 0.655 GoP baseline.
3. **Per-word diagnostic-yield ranking** vs phonetic complexity → seed for designing
   harder diagnostic phrases.
4. Get **more controls** and audit **recording-channel confounds** before any clinical claim.
```
