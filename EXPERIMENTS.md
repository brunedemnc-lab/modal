# Experiment Log

## Baseline

| Model | Frames | Pretrained | Val acc |
|-------|--------|------------|---------|
| CNN-LSTM (ResNet34 + LSTM) | 4 | No | 33.9% |

---

## Changes made

### 1. Bug fix — `create_submission.py`
Removed `import pdb; pdb.set_trace()` that was freezing inference.  
**Impact:** no model effect, mais la génération de soumission fonctionne maintenant.

---

### 2. VideoMAE classifier (Track B) — abandonné pour DINOv2
**Files:** `src/models/videomae_model.py`, `src/configs/model/videomae.yaml`, `src/configs/experiment/track_b_videomae.yaml`

**Architecture:** Video Vision Transformer (ViT-Base, 86M params) pré-entraîné sur SSv2.

**Problème découvert :** VideoMAE a été entraîné avec **16 frames**. Or les données ne contiennent que **4 frames par vidéo**. Avec `num_frames=16`, le dataset répéterait chaque frame 4×, donnant une vidéo statique — l'avantage temporel de VideoMAE disparaît. → **Remplacé par DINOv2 + Temporal Attention (voir change 4).**

---

### 3. Differential learning rates — `train.py`
Ajout de `backbone_lr_scale` dans l'optimizer. Le backbone reçoit `lr × scale`, la tête reçoit `lr`.

**Pourquoi :** Le backbone pré-entraîné a besoin d'un LR faible pour préserver ses représentations (éviter le "catastrophic forgetting"). La tête de classification, initialisée aléatoirement, a besoin d'un LR plus élevé pour apprendre vite.

**Impact sur les anciens experiments :** aucun — `scale=1.0` par défaut.

---

### 4. DINOv2 + Temporal Attention (Track B) — modèle actif
**Files:** `src/models/dinov2_temporal.py`, `src/configs/model/dinov2_temporal.yaml`, `src/configs/experiment/track_b_dinov2.yaml`

**Pourquoi DINOv2 plutôt que VideoMAE pour 4 frames :**
- VideoMAE nécessite 16 frames (tubelet embedding fixe) → incompatible avec nos données
- DINOv2 traite chaque frame indépendamment → fonctionne avec n'importe quel T
- DINOv2 ViT-B/14 offre des features visuelles état de l'art (pré-entraîné sur 142M images)

**Architecture :**
1. DINOv2 ViT-B/14 appliqué à chaque frame → (B, T, 768)
2. Positional embedding temporel appris → encode l'ordre des frames
3. Temporal TransformerEncoder (2 couches, 8 têtes) → raisonnement inter-frames
4. Mean pool + LayerNorm → (B, 768)
5. Linear classifier → (B, 33)

**Comparaison :**

| | CNN-LSTM | VideoMAE | DINOv2 + Temporal Attn |
|--|----------|----------|------------------------|
| Features par frame | ResNet34 scratch | ViT-B SSv2 | ViT-B/14 DINOv2 (142M images) |
| Temporal modeling | LSTM (séquentiel) | Attention 3D jointe | Transformer sur T tokens |
| Compatible 4 frames | Oui | Non (16 requis) | Oui |
| Params | ~21M | ~86M | ~87M |

**Training config:** `lr=1e-4` (tête), `1e-5` (backbone), `batch_size=16`, 10 epochs, `num_frames=4`.

**Run:**
```bash
cd src && python train.py experiment=track_b_dinov2
```

**Val acc:** _à remplir après entraînement_

---

## Résultats

| Modèle | Frames | Val acc |
|--------|--------|---------|
| CNN-LSTM (baseline) | 4 | 33.9% |
| DINOv2 + Temporal Attn | 4 | _à remplir_ |

---

## Idées à tester

| Idée | Track | Gain attendu |
|------|-------|--------------|
| DINOv2-Large (`vit_large_patch14_dinov2`) | B | +2–4% |
| Test-time augmentation (moyenne sur flips/crops) | B | +1–2% |
| Plus de couches temporelles (2 → 4) | B | +1–2% |
| Ensemble CNN-LSTM + DINOv2 | B | +3–5% |
| Plus de frames pour CNN-LSTM (4 → 8) | A | +2–4% |
