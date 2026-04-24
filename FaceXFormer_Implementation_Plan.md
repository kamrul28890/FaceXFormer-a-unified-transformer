# FaceXFormer Reproduction — Detailed Implementation Plan

**Project**: Full 10-task reproduction of FaceXFormer  
**Team**: Preetom Saha Arko & Md Kamruzzaman Kamrul, Purdue University  
**Target**: Paper-accurate architecture (96 task tokens) + 10 tasks including expression + face recognition  
**Cluster**: 8× A100 GPUs, PyTorch DDP, FP16

---

## Current State Assessment

### What your teammate has already built (the 8-task codebase)

| Component | Status | Notes |
|-----------|--------|-------|
| `network/models/facexformer.py` | ✅ Functional | Swin-B backbone, MLP-Fusion (SegFormer-style), FaceDecoder, all 8 task heads |
| `network/models/transformer.py` | ✅ Functional | TwoWayTransformer (TSA → TFCA → FTCA), correct 2-layer depth |
| `datasets.py` | ✅ Functional | 8 datasets: CelebAMask-HQ, 300W, 300W-LP, CelebA, UTKFace, FairFace, COFW; test-only: 300VW, BIWI, LFWA |
| `losses.py` | ✅ Functional | Dice+CE (seg), STAR-approx (lm), Geodesic (pose), BCE (attr/vis), L1+CE (age), CE (gender/race) |
| `train.py` | ✅ Functional | DDP, BalancedMultiTaskBatchSampler, upsampling, LR schedule, checkpointing |
| `config.py` | ✅ Functional | Auto-configured batch sizes, all hyperparams |
| `evaluate.py` | ✅ Functional | Per-dataset, per-task metrics |

### What is missing (gaps to close)

| Gap | Priority | Effort |
|-----|----------|--------|
| **Token count mismatch**: code uses 18 tokens (SAM-style); paper requires 96 (68 LM + 9 pose + 19 seg + others) | **Critical** | Medium |
| **Expression task** (RAF-DB + AffectNet datasets, CE loss, expression head) | **Critical** | Medium |
| **Face recognition task** (MS1MV3 dataset, PartialFC, ArcFace loss, FR head) | **Critical** | High |
| **Hourglass landmark head**: paper uses a stacked hourglass on the 68 tokens, not direct MLP | **High** | Medium |
| **True STAR loss**: current implementation is smooth-L1 approximation, not the actual STARLoss formulation | **High** | Low |
| **Per-task augmentation pipelines** (Appendix F): random rotation ±18°, flip, blur, occlusion, gamma | **High** | Medium |
| **Landmark alignment preprocessing**: 5-point affine transform for 300W/COFW | **Medium** | Low |
| **Loose crop strategy** for head pose (300W-LP) | **Medium** | Low |
| **Inference baseline**: run released checkpoint, confirm paper numbers | **Medium** | Low |
| **Ablation infrastructure**: flag to swap bidirectional ↔ unidirectional cross-attention | **Low** | Low |

---

## Architecture Gap: Token Count Fix (Phase 1, Priority 0)

This is the most impactful change and must be done before any training. The current codebase diverges from the paper fundamentally here.

**Current (wrong)**:
```
landmarks_token: Embedding(1, 256)   → MLP → 136 coords
pose_token:      Embedding(1, 256)   → MLP → 3 angles
mask_tokens:     Embedding(11, 256)  → hypernetwork MLP → segmentation
Total: 18 tokens
```

**Paper specification**:
```
landmarks_tokens: Embedding(68, 256)  → one token per landmark
pose_tokens:      Embedding(9, 256)   → one token per element of 3×3 rotation matrix
seg_tokens:       Embedding(19, 256)  → one token per semantic class
attribute_tokens: Embedding(40, 256)  → one token per binary attribute
age_token:        Embedding(1, 256)
gender_token:     Embedding(1, 256)
race_token:       Embedding(1, 256)
expression_token: Embedding(1, 256)
fr_token:         Embedding(1, 256)
visibility_token: Embedding(1, 256)
Total: 68+9+19+40+1+1+1+1+1+1 = 142 tokens
```

**What changes in the model**:
- `FaceDecoder.__init__`: replace single embeddings with correct token counts
- `FaceDecoder.forward`: extract token slices correctly by index range, not single index
- `landmarks_prediction_head`: changes from `MLP(256→136)` to a **hourglass head** operating on the [B, 68, 256] token set → outputs [B, 68, 2] then flattened
- `pose_prediction_head`: changes from `MLP(256→3)` to `MLP(256→1)` × 9 (one per rotation matrix element), or equivalently take the mean/concat of 9 tokens → [B, 9] → reshape to [B, 3, 3]
- `seg output`: 19 tokens instead of 11 (19 semantic classes in CelebAMask-HQ)
- `attribute_output`: 40 tokens → each token produces a binary logit via `MLP(256→1)` per token, or concat → `MLP(40×256→40)`
- Self-attention (TSA) now operates on 142 tokens — this is fine, it just scales

**Note on segmentation classes**: The paper uses 19 classes for CelebAMask-HQ (background + 18 facial regions). The current code uses 11. You will need to verify which label mapping the paper uses and update `SEGMENTATION_CLASSES` in config accordingly. Check the original FaceXFormer repo's inference code for the exact mapping.

---

## Phase-by-Phase Implementation Plan

---

### Phase 0 — Baseline Verification (Before writing any new code)
**Goal**: Confirm the released pretrained weights reproduce paper numbers. This gives you a reference point and catches environment issues early.

**Steps**:
1. Install the original FaceXFormer inference repo (`pip install -r requirements.txt`)
2. Download pretrained weights from HuggingFace (`kartiknarayan/facexformer`)
3. Run `inference.py` on the provided test images; confirm outputs are sane
4. Run `evaluate.py` (original repo) on your downloaded test sets:
   - CelebAMask-HQ test set → target F1 ≈ 92.01
   - 300W full set → target NME ≈ 4.67
   - BIWI → target MAE ≈ 3.52
5. Document every result in a **gap analysis table** (Paper claim | Reproduced | Gap)

**Deliverable**: `gap_analysis_baseline.csv`

---

### Phase 1 — Architecture Correction (Token Count + Hourglass + Seg Classes)
**Files to modify**: `network/models/facexformer.py`, `config.py`

#### 1a. Fix token counts in `FaceDecoder`

Replace the token embeddings in `FaceDecoder.__init__`:

```python
# Paper-accurate token definitions
self.landmarks_tokens  = nn.Embedding(68, 256)   # one per landmark
self.pose_tokens       = nn.Embedding(9, 256)    # one per rotation matrix element
self.seg_tokens        = nn.Embedding(19, 256)   # one per semantic class (verify count)
self.attribute_tokens  = nn.Embedding(40, 256)   # one per binary attribute
self.age_token         = nn.Embedding(1, 256)
self.gender_token      = nn.Embedding(1, 256)
self.race_token        = nn.Embedding(1, 256)
self.expression_token  = nn.Embedding(1, 256)    # new
self.fr_token          = nn.Embedding(1, 256)    # new
self.visibility_token  = nn.Embedding(1, 256)
```

In `forward`, build the token sequence with explicit offset tracking:
```python
output_tokens = torch.cat([
    self.landmarks_tokens.weight,   # [68, 256]  idx 0–67
    self.pose_tokens.weight,        # [9, 256]   idx 68–76
    self.seg_tokens.weight,         # [19, 256]  idx 77–95
    self.attribute_tokens.weight,   # [40, 256]  idx 96–135
    self.age_token.weight,          # [1, 256]   idx 136
    self.gender_token.weight,       # [1, 256]   idx 137
    self.race_token.weight,         # [1, 256]   idx 138
    self.expression_token.weight,   # [1, 256]   idx 139
    self.fr_token.weight,           # [1, 256]   idx 140
    self.visibility_token.weight,   # [1, 256]   idx 141
], dim=0)  # [142, 256]
```

Then slice correctly after transformer:
```python
lm_tokens_out      = hs[:, 0:68, :]    # [B, 68, 256]
pose_tokens_out    = hs[:, 68:77, :]   # [B, 9, 256]
seg_tokens_out     = hs[:, 77:96, :]   # [B, 19, 256]
attr_tokens_out    = hs[:, 96:136, :]  # [B, 40, 256]
age_token_out      = hs[:, 136, :]     # [B, 256]
gender_token_out   = hs[:, 137, :]     # [B, 256]
race_token_out     = hs[:, 138, :]     # [B, 256]
expr_token_out     = hs[:, 139, :]     # [B, 256]
fr_token_out       = hs[:, 140, :]     # [B, 256]
vis_token_out      = hs[:, 141, :]     # [B, 256]
```

#### 1b. Fix the landmark head — Hourglass

The paper uses a **stacked hourglass network** on the 68 landmark tokens to produce heatmaps, not a direct MLP. This is the most complex architectural addition.

The practical implementation takes the 68 tokens [B, 68, 256] and:
1. Reshape → [B, 256, 8, 8] or similar spatial arrangement (this is the underspecified part — you'll need to make a principled decision and document it)
2. Pass through 2–3 hourglass blocks (encoder-decoder with skip connections)
3. Output [B, 68, H, W] heatmaps → apply softargmax to get [B, 68, 2] coordinates

**Recommended simplification to document**: Since the exact hourglass is not specified in the paper or code release, you can implement a lightweight version — a 3-layer encoder-decoder (Conv → BN → ReLU) applied to the token features — and explicitly note this as an "inferred implementation" in your reproducibility table. A direct MLP baseline should also be kept as an ablation point.

**Minimum viable version** (implement first, upgrade later):
```python
self.landmarks_prediction_head = nn.Sequential(
    nn.Linear(256 * 68, 1024),
    nn.ReLU(),
    nn.Linear(1024, 136)  # 68*2
)
# in forward: lm_tokens_out.flatten(1) → head
```

#### 1c. Fix segmentation output

Update `output_upscaling` and `output_hypernetwork_mlps` to use 19 classes instead of 11. Also update `config.py`:
```python
SEGMENTATION_CLASSES = 19  # was 11
```
And update `DiceLoss`, `CrossEntropyLoss` in `losses.py` to use `ignore_index` correctly with 19 classes. Update all `seg_target = torch.clamp(seg_target, 0, 18)` guards.

#### 1d. Fix pose head

Current: `MLP(256 → 3)` gives Euler angles directly.  
Paper: 9 tokens → one per rotation matrix element → reshape to [B, 3, 3].

```python
self.pose_prediction_head = MLP(256, 256, 1, 3)  # one scalar per token
# in forward:
pose_output = self.pose_prediction_head(pose_tokens_out).squeeze(-1)  # [B, 9]
# loss uses this as flat 9-vector; reshape to [B, 3, 3] inside geodesic loss
```

Update `losses.py` geodesic loss to accept [B, 9] input and reshape internally.

#### 1e. Fix attribute head

Current: single token → MLP(256→40).  
Paper: 40 tokens → each produces one binary logit.

```python
self.attribute_prediction_head = MLP(256, 256, 1, 3)  # scalar per token
# in forward:
attr_output = self.attribute_prediction_head(attr_tokens_out).squeeze(-1)  # [B, 40]
```

This is functionally equivalent in output shape but architecturally different — each attribute gets its own dedicated token that only attends to relevant face regions.

#### 1f. Update config

```python
TASK_TOKENS = {
    'landmark': 68,
    'headpose': 9,
    'segmentation': 19,
    'attribute': 40,
    'age': 1,
    'gender': 1,
    'race': 1,
    'expression': 1,   # new
    'face_recognition': 1,  # new
    'visibility': 1,
}
TOTAL_TASK_TOKENS = 142
```

---

### Phase 2 — Expression Task
**New files**: `datasets.py` additions, `losses.py` additions, `network/models/facexformer.py` additions

#### 2a. Expression datasets

**RAF-DB** (training + evaluation):
- Root: `./datasets/RAF-DB/`
- Structure: `./basic/Image/aligned/` for images, `./basic/list_patition_label.txt` for split+labels
- 7 classes: Surprise, Fear, Disgust, Happy, Sad, Anger, Neutral
- Label file format: `train_00001.jpg 1` (1-indexed)

```python
class RAFDBDataset(TaskDataset):
    TASK_ID = 8  # new task ID
    NUM_CLASSES = 7
    
    def __init__(self, split='train', dataset_root='./datasets'):
        self.data_root = Path(dataset_root) / 'RAF-DB' / 'basic'
        self.img_dir = self.data_root / 'Image' / 'aligned'
        label_file = self.data_root / 'list_patition_label.txt'
        
        self.samples = []
        with open(label_file) as f:
            for line in f:
                fname, label = line.strip().split()
                # train/test split by filename prefix
                if split == 'train' and fname.startswith('train'):
                    self.samples.append((fname, int(label) - 1))  # 0-indexed
                elif split == 'test' and fname.startswith('test'):
                    self.samples.append((fname, int(label) - 1))
    
    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(self.img_dir / fname).convert('RGB')
        img = simple_augmentation(img)
        return img, {
            'task_id': torch.tensor(self.TASK_ID),
            'expression': torch.tensor(label, dtype=torch.long)
        }
```

**AffectNet** (training only — it's very large, ~450K images):
- Root: `./datasets/AffectNet/`
- Use 8-class version (add contempt) or 7-class (drop contempt to match RAF-DB) — **you must decide and document this**. The paper says "7 or 8 emotion classes." Recommended: train on 8, evaluate on 7 (RAF-DB only has 7).
- CSV annotations: `training.csv` / `validation.csv` with columns: `subDirectory_filePath, expression`

```python
class AffectNetDataset(TaskDataset):
    TASK_ID = 8
    NUM_CLASSES = 8  # or 7, document your choice
    
    def __init__(self, split='train', dataset_root='./datasets', num_classes=8):
        self.data_root = Path(dataset_root) / 'AffectNet'
        self.num_classes = num_classes
        csv_file = 'training.csv' if split == 'train' else 'validation.csv'
        df = pd.read_csv(self.data_root / csv_file)
        # Filter to valid class range
        df = df[df['expression'] < num_classes]
        self.samples = list(zip(df['subDirectory_filePath'], df['expression']))
    
    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img = Image.open(self.data_root / rel_path).convert('RGB')
        img = simple_augmentation(img)
        return img, {
            'task_id': torch.tensor(self.TASK_ID),
            'expression': torch.tensor(int(label), dtype=torch.long)
        }
```

#### 2b. Expression loss

Add to `losses.py` in `MultiTaskLoss.forward`:
```python
# EXPRESSION LOSS (CE)
if 'expression_output' in predictions and 'expression' in labels:
    expr_mask = (task_ids == 8)
    if expr_mask.any():
        expr_pred_masked = predictions['expression_output'][expr_mask]
        expr_target = labels['expression'][expr_mask].to(device).long()
        expr_loss = F.cross_entropy(expr_pred_masked, expr_target)
        losses['exp'] = expr_loss
        total_loss += self.loss_weights['exp'] * expr_loss
```

Add `'exp': 1.0` to `LOSS_WEIGHTS` in config.

#### 2c. Expression head in model

The `expression_token` is already added in Phase 1. Add the head:
```python
self.expression_prediction_head = MLP(256, 256, 8, 3)  # or 7
```
In `FaceDecoder.forward`:
```python
expression_output = self.expression_prediction_head(expr_token_out)  # [B, 8]
```

#### 2d. Update `train.py` forward pass

The forward returns and predictions dict need to add `expression_output`. Update:
- Model forward signature: add `expression_out` to return tuple
- `predictions` dict: add `'expression_output': expression_out`
- `collate_fn` dummy values: add `'expression': torch.tensor(0, dtype=torch.long)`
- `train_datasets` dict: add `'expression': [rafdb_train, affectnet_train]`
- `test_datasets` dict: add `'expression': [rafdb_test]`
- Task ID map: add `'expression': 8`

---

### Phase 3 — Face Recognition Task
**This is the most engineering-intensive addition.**

#### 3a. MS1MV3 dataset

MS1MV3 is distributed in MXNet `.rec` format. You have two options:
- **Option A (Recommended)**: Convert to image folder format using the `rec2image` script from InsightFace (avoids MXNet dependency at training time)
- **Option B**: Use `mxnet.recordio` reader directly at training time

For Option A (recommended), run this once during dataset prep:
```bash
python tools/convert_ms1mv3.py \
    --rec_path ./datasets/MS1MV3/train.rec \
    --idx_path ./datasets/MS1MV3/train.idx \
    --output_dir ./datasets/MS1MV3/images/
```

The dataset class is simpler after conversion:
```python
class MS1MV3Dataset(TaskDataset):
    TASK_ID = 9
    
    def __init__(self, split='train', dataset_root='./datasets'):
        self.img_root = Path(dataset_root) / 'MS1MV3' / 'images'
        # Build identity-to-index mapping
        self.samples = []  # list of (img_path, identity_idx)
        self.num_identities = 0
        
        for identity_dir in sorted(self.img_root.iterdir()):
            if identity_dir.is_dir():
                identity_idx = self.num_identities
                for img_path in identity_dir.glob('*.jpg'):
                    self.samples.append((img_path, identity_idx))
                self.num_identities += 1
        
        # MS1MV3 has ~93K identities, ~5.1M images
        # Only use train split (no standard test set — evaluation is on LFW/CFP-FP/etc.)
    
    def __getitem__(self, idx):
        img_path, identity_idx = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = simple_augmentation(img)
        return img, {
            'task_id': torch.tensor(self.TASK_ID),
            'face_recognition': torch.tensor(identity_idx, dtype=torch.long)
        }
```

#### 3b. PartialFC + ArcFace loss

This is architecturally special. PartialFC is not a loss you add to `losses.py` in the normal way — it's a **distributed classifier** that maintains a partial subset of the full identity embedding matrix on each GPU.

**Minimum viable implementation** (document as "simplified ArcFace without PartialFC" — valid for smaller-scale):
```python
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_dim=256, num_identities=93000, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_identities, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        # L2 normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings_norm, weight_norm)  # [B, N_identities]
        
        # ArcFace margin
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        
        # Only apply margin to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.s
        
        return F.cross_entropy(output, labels)
```

**Full PartialFC** (if you want to match the paper exactly): Use the InsightFace `partial_fc` module. It requires distributed communication to maintain a subset of the classifier on each GPU. This adds significant complexity. Implementation steps:
1. `pip install insightface`
2. Replace `ArcFaceLoss` with `insightface.recognition.arcface_torch.losses.PartialFC`
3. Pass `num_class=93000, sample_rate=0.1` (sample 10% of identities per batch)
4. The PartialFC module requires its own optimizer step — it is not part of the main model's parameter group

**Recommended approach**: Implement simplified ArcFace first and get it training. Then swap to PartialFC if memory is a concern (it is necessary for 93K identities on a single forward pass — the weight matrix alone is 93000 × 256 × 4 bytes ≈ 95MB per GPU, which is manageable on A100s).

#### 3c. Face recognition head and evaluation

The FR head outputs an **embedding** [B, 256], not class logits. The ArcFace loss uses this embedding against the identity classifier. At evaluation time, you compute verification accuracy (is pair A/B the same person?) not classification accuracy.

```python
self.fr_prediction_head = MLP(256, 256, 256, 3)  # embedding output
```

**Evaluation datasets**: LFW, CFP-FP, AgeDB, CALFW, CPLFW — these are all **verification** datasets with pair lists.

Add to `evaluate.py`:
```python
def compute_verification_accuracy(embeddings_a, embeddings_b, labels, threshold=None):
    # Cosine similarity
    sim = F.cosine_similarity(embeddings_a, embeddings_b)
    if threshold is None:
        # Find best threshold on the pair list
        best_acc, best_thresh = 0, 0
        for t in np.arange(0.1, 1.0, 0.01):
            pred = (sim > t).float()
            acc = (pred == labels).float().mean()
            if acc > best_acc:
                best_acc, best_thresh = acc, t
        return best_acc
    return (sim > threshold).float().eq(labels).float().mean()
```

You will need dataset classes for each verification set. Each follows the same pattern: load pair list (person1/img1, person2/img2, same/different), run model on both images, compare embeddings.

---

### Phase 4 — Augmentation Pipelines (Appendix F)
**File to modify**: `datasets.py`

The current `simple_augmentation` only resizes and normalizes. You need task-specific augmentation. Create a proper augmentation module:

```python
import torchvision.transforms.functional as TF
import random

class TaskAugmentation:
    def __init__(self, task_name):
        self.task_name = task_name
    
    def __call__(self, image, landmarks=None):
        # Paper Appendix F probabilities:
        
        # Random horizontal flip (50%) — applies to all tasks
        # NOTE: for landmarks, must mirror x-coordinates and swap symmetric pairs
        if random.random() < 0.5:
            image = TF.hflip(image)
            if landmarks is not None:
                W = image.width
                landmarks[:, 0] = W - landmarks[:, 0]
                landmarks = self._swap_symmetric_landmarks(landmarks)
        
        # Random rotation ±18° (all tasks except head pose)
        if self.task_name != 'headpose' and random.random() < 0.5:
            angle = random.uniform(-18, 18)
            image = TF.rotate(image, angle)
            if landmarks is not None:
                landmarks = self._rotate_landmarks(landmarks, angle, image.width, image.height)
        
        # Random scaling ±10%
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            image = image.resize((new_w, new_h))
        
        # Gaussian blur (all tasks, p=0.2)
        if random.random() < 0.2:
            sigma = random.uniform(0.1, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Grayscale conversion (p=0.1)
        if random.random() < 0.1:
            image = image.convert('L').convert('RGB')
        
        # Gamma adjustment (p=0.3)
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.3)
            image = TF.adjust_gamma(image, gamma)
        
        # Occlusion (random rectangle, p=0.2) — NOT for landmarks or visibility
        if self.task_name not in ['landmark', 'visibility'] and random.random() < 0.2:
            image = self._apply_occlusion(image)
        
        # Final resize to 224×224
        image = image.resize((224, 224), Image.BILINEAR)
        
        # Normalize (ImageNet stats)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        
        return image, landmarks
```

**Important edge cases to handle**:
- Landmark flip requires swapping symmetric pairs (left eye ↔ right eye, etc.) — there is a standard 68-point symmetry map for this
- Head pose should NOT use rotation augmentation (it would corrupt the ground truth angles)
- Segmentation masks must undergo the same spatial transforms as images
- COFW visibility: augmentation must be applied consistently to both image and visibility mask

---

### Phase 5 — True STAR Loss
**File to modify**: `losses.py`

The current implementation is smooth-L1, not STARLoss. The actual STARLoss from Zhou et al. (CVPR 2023) adds a **self-adaptive ambiguity reduction** term that weights each landmark by its estimated uncertainty:

```python
class STARLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        pred:   [B, 68, 2] predicted coordinates
        target: [B, 68, 2] ground truth coordinates
        """
        # Euclidean distance per landmark
        diff = pred - target  # [B, 68, 2]
        dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)  # [B, 68]
        
        # Self-adaptive weight: landmarks with higher variance get lower weight
        # Variance estimated across batch for each landmark
        weight = 1.0 / (dist.detach().std(dim=0, keepdim=True) + 1e-6)  # [1, 68]
        weight = weight / weight.sum() * 68  # normalize so mean weight = 1
        
        loss = (weight * dist).mean()
        return loss
```

This is still an approximation of the full STAR formulation (which uses a learnable per-landmark uncertainty). Document this in your reproducibility table.

---

### Phase 6 — Dataset Path Cleanup & Collate Function Updates
**File to modify**: `datasets.py`, `train.py`

Currently all dataset roots point to `'../facexformer-my/datasets'`. Change to `'./datasets'` (or parameterize via config/argparse) so the code runs from the project root without assuming a sibling directory.

Update the `multi_task_collate_fn` dummy values for the new tasks:
```python
elif key == 'expression':
    values.append(torch.tensor(0, dtype=torch.long))
elif key == 'face_recognition':
    values.append(torch.tensor(0, dtype=torch.long))
```

Update task ID mapping consistently across all files:
```python
TASK_ID_MAP = {
    'segmentation': 0,
    'landmark': 1,
    'headpose': 2,
    'attribute': 3,
    'age': 4,
    'gender': 5,
    'race': 6,
    'visibility': 7,
    'expression': 8,    # new
    'face_recognition': 9  # new
}
```

---

### Phase 7 — Staged Training (As Proposed)
**Files**: `train.py` additions, SLURM scripts

Run training in stages, using each checkpoint to initialize the next:

| Stage | Tasks | Datasets | Target |
|-------|-------|----------|--------|
| Stage 1 | Seg + LM + Pose | CelebAMask-HQ, 300W, 300W-LP | Validate sampler + loss pipeline |
| Stage 2 | + Attr + Age/Gender/Race | + CelebA, UTKFace, FairFace | Validate classification heads |
| Stage 3 | + Visibility + Expression | + COFW, RAF-DB, AffectNet | 8-task checkpoint |
| Stage 4 | + Face Recognition | + MS1MV3 | Full 10-task checkpoint |

**SLURM launch script** (8× A100):
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=320G

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train.py \
    --stage 4 \
    --resume checkpoints/stage3_best.pth
```

---

### Phase 8 — Evaluation & Ablations

#### 8a. Per-task evaluation targets (from paper)

| Task | Dataset | Metric | Paper Target |
|------|---------|--------|-------------|
| Face Parsing | CelebAMask-HQ | F1 | 92.01 |
| Landmarks | 300W Full | NME | 4.67 |
| Head Pose | BIWI | MAE | 3.52 |
| Attributes | CelebA | Accuracy | 91.83% |
| Age | UTKFace | MAE | 4.17 years |
| Visibility | COFW | Recall@80%P | 72.56 |
| Expression | RAF-DB | Accuracy | 88.24% |
| Face Recognition | LFW/CFP-FP/AgeDB/CALFW/CPLFW | Mean Acc | 95.94% |

#### 8b. Ablation infrastructure

Add a flag to `config.py` and `transformer.py` to swap attention type:

```python
# config.py
ATTENTION_TYPE = 'bidirectional'  # or 'unidirectional'
```

In `TwoWayAttentionBlock.forward`, wrap FTCA in a conditional:
```python
if config.ATTENTION_TYPE == 'bidirectional':
    attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
    keys = keys + attn_out
    keys = self.norm4(keys)
```

This lets you reproduce Table 7 ablation (a) with a single config change.

---

## File-by-File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `network/models/facexformer.py` | **Rewrite FaceDecoder** | 142 tokens, correct head architectures, expression + FR heads |
| `network/models/transformer.py` | Minor | Add attention type flag for ablation |
| `config.py` | Updates | 142 tokens, 19 seg classes, add expression/FR loss weights and task IDs |
| `datasets.py` | **Add classes** | RAFDBDataset, AffectNetDataset, MS1MV3Dataset, LFW/CFP-FP/AgeDB/CALFW/CPLFW verification datasets; fix dataset root paths; add TaskAugmentation class |
| `losses.py` | **Add + fix** | True STARLoss, ArcFaceLoss, expression CE loss; update pose loss for [B,9] input; update seg loss for 19 classes |
| `train.py` | Updates | Add expression + FR to forward pass, predictions dict, train/test dataset dicts, task ID map; add stage flag |
| `evaluate.py` | **Add** | Verification accuracy metric, FR evaluation loop, expression evaluation loop |
| `tools/convert_ms1mv3.py` | **New file** | One-time MXNet rec → image folder converter |

---

## Open Decisions You Must Document (Reproducibility Table)

These are paper underspecifications that require a judgment call. Record each one explicitly:

| # | Issue | Paper Says | Your Decision | Sensitivity |
|---|-------|------------|---------------|-------------|
| 1 | All λ loss weights | Omitted | Start at 1.0, tune via grad-norm balancing | High |
| 2 | Hourglass landmark head details | "hourglass network" | Lightweight 3-layer encoder-decoder on 68 tokens | Medium |
| 3 | Expression classes: 7 vs 8 | "7 or 8 emotion classes" | Train AffectNet with 8, evaluate RAF-DB with 7 | Medium |
| 4 | PartialFC vs standard ArcFace | PartialFC with ArcFace | Start with simplified ArcFace, upgrade if needed | High |
| 5 | Exact segmentation class mapping | 19 classes implied | Verify against original inference code | High |
| 6 | Landmark flip symmetry map | Not specified | Standard 68-point ibug symmetry map | Low |
| 7 | Age bin centers for L1 loss | Not specified | [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 75.0] | Low |
| 8 | LR for 8 vs 10 task stages | 10−4 for full system | Scale proportionally per staged initialization | Medium |

---

## Recommended Work Division

Given you are two people:

**Person A (architecture + training infrastructure)**:
- Phase 1 (token count fix + all head corrections)
- Phase 5 (STARLoss)
- Phase 7 (staged training scripts + SLURM)
- Phase 8b (ablation flags)

**Person B (datasets + losses)**:
- Phase 0 (baseline verification)
- Phase 2 (expression datasets + loss + head wiring)
- Phase 3 (MS1MV3 dataset + ArcFace loss + verification eval)
- Phase 4 (augmentation pipelines)
- Phase 6 (dataset path fixes + collate updates)

Both phases 1 and 3 depend on the token count being fixed first — make sure Phase 1 is merged and tested before Person B starts wiring expression/FR into the forward pass.

---

## Critical Path (What Blocks What)

```
Phase 0 (baseline)
    └── parallel with everything else

Phase 1 (token fix)  ← MUST complete first
    ├── Phase 2 (expression)
    ├── Phase 3 (face recognition)
    └── Phase 4 (augmentation) ← can start in parallel with Phase 1

Phase 2 + Phase 3 + Phase 4 + Phase 5 + Phase 6 ← all parallel

Phase 7 Stage 1 (3-task train)
    └── Phase 7 Stage 2 (6-task)
        └── Phase 7 Stage 3 (8-task + expression)
            └── Phase 7 Stage 4 (10-task + FR)

Phase 8 (evaluation + ablations) ← after Stage 4
```

---

## Quick-Start Checklist Before Training

- [ ] Verify dataset directory structure matches dataset class expectations
- [ ] Confirm CelebAMask-HQ has 19 class labels (not 11) — check the label files
- [ ] Run `python test_setup.py` (already exists) — make sure all imports pass
- [ ] Run `python losses.py` — confirms loss functions run on dummy data
- [ ] Run `python -c "from datasets import *; d = CelebAMaskHQDataset('train'); print(d[0])"` — confirm data loads
- [ ] Verify MS1MV3 conversion completed (check identity count ≈ 93,000)
- [ ] Test single-GPU run with `train_simple.py` for 10 batches before DDP launch
- [ ] Confirm SLURM environment has `NCCL_IB_DISABLE=1` if InfiniBand is not available

---

*Plan version 1.0 — April 2026*  
*Based on codebase audit of facexformer-main-main.zip and proposal document*
