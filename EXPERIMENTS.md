# Experiment Log

## Baseline

| Model | Frames | Pretrained | Val acc |
|-------|--------|------------|---------|
| CNN-LSTM (ResNet34 + LSTM) | 4 | No | 33.9% |

---

## Changes made

### 1. Bug fix — `create_submission.py`
Removed `import pdb; pdb.set_trace()` that was freezing inference.  
**Impact:** no model effect, but submission generation now works.

---

### 2. VideoMAE classifier (Track B)
**Files:** `src/models/videomae_model.py`, `src/configs/model/videomae.yaml`, `src/configs/experiment/track_b_videomae.yaml`

**Architecture:** Video Vision Transformer (ViT-Base, 86M params) from HuggingFace.  
Pretrained with masked autoencoding on Something-Something V2 — the same dataset as this challenge.

**Key differences vs CNN-LSTM:**

| | CNN-LSTM | VideoMAE |
|--|----------|----------|
| Temporal modeling | LSTM (sequential, hidden state) | Full spatio-temporal attention (every patch attends to all others) |
| Frame features | ResNet34 from scratch | ViT-Base pretrained on SSv2 |
| Frames | 4 | 16 |
| Parameters | ~21M | ~86M |

**Training config:** `lr=1e-4` (head), `1e-5` (backbone via `backbone_lr_scale=0.1`), `batch_size=8`, 15 epochs.

**Run:**
```bash
cd src && python train.py experiment=track_b_videomae
```

**Val acc:** _fill after training_

---

### 3. Differential learning rates — `train.py`
Added `backbone_lr_scale` in the optimizer. Backbone gets `lr × scale`, head gets `lr`.

**Why:** The pretrained backbone needs a small LR to preserve its representations. The new random head needs a large LR to learn quickly. Using one LR for both either destroys the backbone or starves the head.

**Impact on existing experiments:** none — defaults to `scale=1.0`.

---

## Next steps

| Idea | Track | Expected gain |
|------|-------|---------------|
| VideoMAE-Large (`videomae-large-finetuned-ssv2`) | B | +2–5% |
| More frames for CNN-LSTM (4 → 8) | A | +2–4% |
| Test-time augmentation (average flipped predictions) | Both | +1–2% |
| Ensemble CNN-LSTM + VideoMAE | B | +3–5% |
