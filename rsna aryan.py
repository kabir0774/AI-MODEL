"""
RSNA KNEE ABNORMALITY DETECTION — merged local pipeline
Merged from: config.py, dicom_utils.py, preprocessing.py, embed.py, model.py,
             dataset.py, train.py, kaggle_pipeline.py

WHAT WAS CHANGED FROM THE ORIGINAL 6 FILES (so nothing is a silent surprise):
  1. TARGETS — the original file had 12 placeholder names ("acl_tear",
     "pcl_tear", ...). Replaced with the real 12 competition targets
     ("ACL", "MCL", "Medial Meniscus", ... "Fracture").
  2. Paths/columns — the original assumed train/<study_id>/<series>/*.dcm,
     train_labels.csv, train_reports.csv, a "study_id" column. Our real
     data is train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm,
     train.csv (StudyInstanceUID + Report + 12 label cols), train_series.csv
     (StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive, Fat_Suppression,
     Anatomical_Plane). Every reference was rewired to match.
  3. Plane/fluid/fat detection — the original guessed these from
     SeriesDescription keywords. We already have the OFFICIAL values for
     every series in train_series.csv/test_series.csv, so scan_series()
     now looks them up directly instead of guessing. More accurate, and
     avoids the multilingual-description problem entirely.
  4. Restricted to the 58 labeled studies — the original mixed in a
     weak-labeled (report-parser) training branch that would need DICOM
     access to ~4400 more study folders. Per your instruction, that branch
     is disabled here: scan_study() is only ever called for the 58 studies
     that have complete official labels in train.csv, and training never
     touches any other study folder. (The weak-label branch is left in as
     dead code behind USE_WEAK_LABELS=False if you want it back later.)
  5. Everything else — slot assignment, the 20th-80th percentile slice
     band, the laterality flip, the attention model architecture, the
     5-fold GroupKFold training loop, the fold-averaged inference — is
     kept exactly as designed in the original files.

Usage:
    python rsna_local_pipeline.py            # trains 5 folds, then predicts test set
    python rsna_local_pipeline.py --train-only
    python rsna_local_pipeline.py --predict-only
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import AutoModel, AutoProcessor

# ============================================================================
# STEP 0 (was config.py) — paths, targets, hyperparameters
# ============================================================================
ON_KAGGLE  = os.path.exists("/kaggle/input")
IS_SERVER  = os.path.isdir("/home/harleen_ece/rsna_knee_ai") and not ON_KAGGLE

if ON_KAGGLE:
    def _find_data_root():
        comp_base = "/kaggle/input/competitions"
        if os.path.isdir(comp_base):
            for d in sorted(os.listdir(comp_base)):
                cand = os.path.join(comp_base, d)
                if os.path.isfile(os.path.join(cand, "train.csv")):
                    return cand
        direct = "/kaggle/input/rsna-knee-abnormality-detection"
        if os.path.isdir(direct):
            return direct
        raise RuntimeError("Competition data not found under /kaggle/input")

    def _find_medsiglip():
        candidates = [
            "/kaggle/input/datasets/kabirverma01/medsiglip/MedSigLIP",
            "/kaggle/input/medsiglip/MedSigLIP",
            "/kaggle/input/medsiglip",
        ]
        for cand in candidates:
            if os.path.isfile(os.path.join(cand, "config.json")):
                return cand
        for root, dirs, files in os.walk("/kaggle/input"):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            if "config.json" in files and "model.safetensors" in files:
                return root
        raise RuntimeError("MedSigLIP dataset not attached")

    DATA_DIR       = _find_data_root()
    MEDSIGLIP_PATH = _find_medsiglip()
    OUTPUT_DIR     = "/kaggle/working"

elif IS_SERVER:
    # SSH server: harleen_ece@cse
    DATA_DIR       = "/home/harleen_ece/rsna_knee_ai/DATA"
    MEDSIGLIP_PATH = "/home/harleen_ece/rsna_knee_ai/MedSigLIP"
    OUTPUT_DIR     = "/home/harleen_ece/rsna_knee_ai/rsna_aryan_ssh"

else:
    # LOCAL PC paths
    DATA_DIR       = os.environ.get("RSNA_DATA_DIR", r"C:\kabir\RSNA\rsna_inference\DATA")
    MEDSIGLIP_PATH = os.environ.get("MEDSIGLIP_PATH", r"C:\kabir\RSNA_Knee_AI\MedSigLIP")
    OUTPUT_DIR     = os.environ.get("RSNA_OUTPUT_DIR", r"C:\kabir\RSNA\rsna_aryan_run")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# real folder/file names (train_series, not train; *_series.csv for the
# official plane/fluid/fat metadata; StudyInstanceUID as the id column)
TRAIN_DICOM_DIR       = os.path.join(DATA_DIR, "train_series")
TEST_DICOM_DIR        = os.path.join(DATA_DIR, "test_series")
TRAIN_CSV             = os.path.join(DATA_DIR, "train.csv")            # StudyInstanceUID, Report, 12 label cols
TRAIN_SERIES_META_CSV = os.path.join(DATA_DIR, "train_series.csv")     # official plane/fluid/fat per series
TEST_SERIES_META_CSV  = os.path.join(DATA_DIR, "test_series.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV              = os.path.join(DATA_DIR, "test.csv")

# Weak labels — parsed from radiology reports, all 4407 studies.
# On SSH server upload final_labels_real_plus_generated.csv to AI-MODEL/
USE_WEAK_LABELS   = True
N_TOTAL_STUDIES   = 4349   # 58 real + 4291 weak (all available)

# Label CSV paths (auto-detected)
def _find_parsed_labels():
    candidates = [
        # SSH server
        "/home/harleen_ece/rsna_knee_ai/AI-MODEL/final_labels_real_plus_generated.csv",
        "/home/harleen_ece/rsna_knee_ai/AI-MODEL/final_labels_real_plus_generated best.csv",
        # Kaggle confirmed paths
        "/kaggle/input/datasets/kabirverma01/final-labels-real-plus-generated-best/final_labels_real_plus_generated best.csv",
        "/kaggle/input/final-labels-real-plus-generated-best/final_labels_real_plus_generated best.csv",
        "/kaggle/input/datasets/kabirverma01/final-labels-real-plus-generated/final_labels_real_plus_generated.csv",
        # local fallback
        r"C:\kabir\RSNA_Knee_AI\parser\output\final_labels_real_plus_generated.csv",
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # scan /kaggle/input as last resort
    if os.path.isdir("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            for f in files:
                if "final_labels_real_plus_generated" in f and f.endswith(".csv"):
                    return os.path.join(root, f)
    return None

PARSED_LABELS_CSV = _find_parsed_labels()
if PARSED_LABELS_CSV:
    print(f"  Parsed labels   : {PARSED_LABELS_CSV}")
else:
    print("  [WARN] Parsed labels CSV not found — will train on 58 official labels only")
    USE_WEAK_LABELS = False

# ---------------------------------------------------------------------------
# The 6 series "slots" every study gets bucketed into (Step 2)
# ---------------------------------------------------------------------------
SLOTS = [
    "SAG_FLUID_FS",    # Sagittal, fluid-sensitive, fat-suppressed (e.g. PD/T2 FS)
    "COR_FLUID_FS",    # Coronal, fluid-sensitive, fat-suppressed
    "AX_FLUID_FS",     # Axial, fluid-sensitive, fat-suppressed
    "SAG_FLUID_NOFS",  # Sagittal, fluid-sensitive, no fat-sat
    "COR_T1",          # Coronal T1
    "SAG_T1",          # Sagittal T1
]
NUM_SLOTS = len(SLOTS)

# Max slices sampled per series, drawn from the 20th-80th percentile band
# (avoids the joint-line ends where there's rarely pathology).
SLICES_PER_SERIES = 12
BAND_LOW, BAND_HIGH = 0.20, 0.80

# The real 12 competition targets (replaces the original file's placeholders)
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]
NUM_TARGETS = len(TARGETS)
assert NUM_TARGETS == 12

# ---------------------------------------------------------------------------
# Model / embedding dims
# ---------------------------------------------------------------------------
EMBED_DIM = 1152      # MedSigLIP image embedding dim
PROJ_DIM = 256         # after the linear projection (Step 4, Stage 1)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 8
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
NUM_CPU_THREADS = 4
MAX_RAM_GB = 30

torch.manual_seed(SEED)
torch.set_num_threads(NUM_CPU_THREADS)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# STEP 1 (was dicom_utils.py) — DICOM scanning
# Plane/fluid/fat now come from the OFFICIAL train_series.csv /
# test_series.csv lookup instead of guessing from SeriesDescription text.
# Laterality still comes from the DICOM tag / description text, as in the
# original file (no official column for it in the metadata CSVs).
# ============================================================================
@dataclass
class SeriesMeta:
    series_dir: str
    series_uid: str
    plane: str          # "SAG" | "COR" | "AX" | "UNKNOWN"
    fluid_sensitive: bool
    fat_suppressed: bool
    laterality: str      # "L" | "R" | "UNKNOWN"
    num_slices: int


_PLANE_MAP = {"Sagittal": "SAG", "Coronal": "COR", "Axial": "AX"}


def _load_series_meta_lookup(csv_path: str) -> dict:
    """(StudyInstanceUID, SeriesInstanceUID) -> (plane, fluid, fat) from the
    official *_series.csv. Empty dict (all-UNKNOWN fallback) if missing."""
    if not os.path.isfile(csv_path):
        print(f"[dicom_utils] WARNING: {csv_path} not found — plane/fluid/fat will be UNKNOWN")
        return {}
    df = pd.read_csv(csv_path, dtype=str)
    df["Fluid_Sensitive"] = df["Fluid_Sensitive"].astype(int).astype(bool)
    df["Fat_Suppression"] = df["Fat_Suppression"].astype(int).astype(bool)
    lookup = {}
    for r in df.itertuples(index=False):
        lookup[(r.StudyInstanceUID, r.SeriesInstanceUID)] = (
            _PLANE_MAP.get(r.Anatomical_Plane, "UNKNOWN"),
            r.Fluid_Sensitive,
            r.Fat_Suppression,
        )
    return lookup


def _infer_laterality(ds) -> str:
    lat = getattr(ds, "Laterality", None) or getattr(ds, "ImageLaterality", None)
    if lat in ("L", "R"):
        return lat
    desc = (getattr(ds, "SeriesDescription", "") or getattr(ds, "StudyDescription", "")).lower()
    if "left" in desc:
        return "L"
    if "right" in desc:
        return "R"
    return "UNKNOWN"


def scan_series(study_uid: str, series_dir: str, meta_lookup: dict) -> SeriesMeta:
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {series_dir}")

    series_uid = os.path.basename(series_dir.rstrip(os.sep))

    # Header-only read on the first file — fast, no pixel data.
    ds = pydicom.dcmread(files[0], stop_before_pixels=True)

    plane, fluid, fat_sup = meta_lookup.get((study_uid, series_uid), ("UNKNOWN", False, False))
    laterality = _infer_laterality(ds)

    return SeriesMeta(
        series_dir=series_dir,
        series_uid=series_uid,
        plane=plane,
        fluid_sensitive=fluid,
        fat_suppressed=fat_sup,
        laterality=laterality,
        num_slices=len(files),
    )


def scan_study(study_uid: str, study_dir: str, meta_lookup: dict) -> list[SeriesMeta]:
    """A study directory holds one subfolder per series."""
    if not os.path.isdir(study_dir):
        print(f"[dicom_utils] skipping {study_dir}: not found")
        return []
    series_dirs = [
        os.path.join(study_dir, d)
        for d in sorted(os.listdir(study_dir))
        if os.path.isdir(os.path.join(study_dir, d))
    ]
    metas = []
    for sd in series_dirs:
        try:
            metas.append(scan_series(study_uid, sd, meta_lookup))
        except Exception as e:  # noqa: BLE001 — keep scanning other series if one is malformed
            print(f"[dicom_utils] skipping {sd}: {e}")
    return metas


# ============================================================================
# STEP 2 (was preprocessing.py) — slot assignment & preprocessing
# Unchanged from the original file, aside from using the corrected
# SeriesMeta fields above.
# ============================================================================
def assign_slot(meta: SeriesMeta) -> str | None:
    """Map (plane, fluid_sensitive, fat_suppressed) -> one of SLOTS.
    Returns None if the series doesn't match any slot (e.g. a localizer)."""
    if meta.plane == "SAG" and meta.fluid_sensitive and meta.fat_suppressed:
        return "SAG_FLUID_FS"
    if meta.plane == "COR" and meta.fluid_sensitive and meta.fat_suppressed:
        return "COR_FLUID_FS"
    if meta.plane == "AX" and meta.fluid_sensitive and meta.fat_suppressed:
        return "AX_FLUID_FS"
    if meta.plane == "SAG" and meta.fluid_sensitive and not meta.fat_suppressed:
        return "SAG_FLUID_NOFS"
    if meta.plane == "COR" and not meta.fluid_sensitive:
        return "COR_T1"
    if meta.plane == "SAG" and not meta.fluid_sensitive:
        return "SAG_T1"
    return None


def assign_slots_for_study(metas: list[SeriesMeta]) -> dict[str, SeriesMeta]:
    """One series per slot. If multiple series match the same slot, keep
    the one with more slices (usually the more complete acquisition)."""
    slot_map: dict[str, SeriesMeta] = {}
    for m in metas:
        slot = assign_slot(m)
        if slot is None:
            continue
        if slot not in slot_map or m.num_slices > slot_map[slot].num_slices:
            slot_map[slot] = m
    return slot_map


def sample_slice_band(num_slices: int) -> np.ndarray:
    """Indices for up to SLICES_PER_SERIES slices, evenly spaced within the
    20th-80th percentile band of the volume."""
    lo = int(round(num_slices * BAND_LOW))
    hi = int(round(num_slices * BAND_HIGH))
    hi = max(hi, lo + 1)
    n = min(SLICES_PER_SERIES, hi - lo)
    return np.linspace(lo, hi - 1, n).round().astype(int)


def load_series_slices(meta: SeriesMeta, flip_lr: bool) -> list[Image.Image]:
    """Load the sampled slices for one series as 2D PIL images, applying
    the laterality flip so every knee is presented in the same orientation."""
    files = sorted(glob.glob(os.path.join(meta.series_dir, "*.dcm")))
    idxs = sample_slice_band(len(files))

    slices = []
    for i in idxs:
        ds = pydicom.dcmread(files[i])
        arr = ds.pixel_array.astype(np.float32)
        arr = 255 * (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        if flip_lr:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        slices.append(img)
    return slices


def preprocess_study(study_metas: list[SeriesMeta]) -> dict[str, list[Image.Image]]:
    """Returns {slot_name: [PIL images]} for whichever slots this study has.
    Missing slots simply aren't in the dict — the presence mask is derived
    from this downstream, in embed_study()."""
    slot_map = assign_slots_for_study(study_metas)
    out = {}
    for slot, meta in slot_map.items():
        flip = meta.laterality == "R"
        out[slot] = load_series_slices(meta, flip_lr=flip)
    return out


# ============================================================================
# STEP 3 (was embed.py) — embedding via MedSigLIP
# Unchanged from the original file, aside from using MEDSIGLIP_PATH above.
# ============================================================================
_medsiglip_model = None
_medsiglip_processor = None


def _load_medsiglip():
    global _medsiglip_model, _medsiglip_processor
    if _medsiglip_model is None:
        print(f"[embed] Loading MedSigLIP from {MEDSIGLIP_PATH} (device={DEVICE})")
        _medsiglip_processor = AutoProcessor.from_pretrained(MEDSIGLIP_PATH)
        _medsiglip_model = AutoModel.from_pretrained(MEDSIGLIP_PATH).to(DEVICE).eval()
    return _medsiglip_model, _medsiglip_processor


@torch.no_grad()
def embed_slices(images: list[Image.Image]) -> torch.Tensor:
    """[len(images), EMBED_DIM] image embeddings, frozen (no grad)."""
    if not images:
        return torch.zeros(0, EMBED_DIM)
    model, processor = _load_medsiglip()
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)
    feats = model.get_image_features(**inputs)  # [N, EMBED_DIM]
    return feats.float().cpu()


def embed_study(slot_slices: dict[str, list[Image.Image]]):
    """slot_slices: {slot_name: [PIL images]} from preprocess_study().
    Returns (embeddings, slot_mask):
      embeddings: [NUM_SLOTS, SLICES_PER_SERIES, EMBED_DIM]
      slot_mask:  [NUM_SLOTS]  (1 if that slot was present, 0 if missing)"""
    embeddings = torch.zeros(NUM_SLOTS, SLICES_PER_SERIES, EMBED_DIM)
    slot_mask = torch.zeros(NUM_SLOTS)

    for slot_idx, slot_name in enumerate(SLOTS):
        images = slot_slices.get(slot_name)
        if not images:
            continue
        feats = embed_slices(images)  # [n_slices, EMBED_DIM]
        n = feats.shape[0]
        embeddings[slot_idx, :n] = feats
        slot_mask[slot_idx] = 1.0

    return embeddings, slot_mask


# ============================================================================
# STEP 4 (was model.py) — disease-specific attention model
# Unchanged from the original file.
# ============================================================================
class DiseaseAttentionModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        proj_dim: int = PROJ_DIM,
        num_slots: int = NUM_SLOTS,
        slices_per_slot: int = SLICES_PER_SERIES,
        num_targets: int = NUM_TARGETS,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slices_per_slot = slices_per_slot
        self.num_targets = num_targets

        # Stage 1
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # Stage 2 — one attention scorer per disease, sharing the projected
        # features. Implemented as a single [proj_dim, num_targets] matrix
        # so it's one matmul instead of 12 separate linear layers.
        self.attn_scorer = nn.Linear(proj_dim, num_targets)

        # Stage 3 — 12 independent binary heads
        self.classifiers = nn.ModuleList(
            [nn.Linear(proj_dim, 1) for _ in range(num_targets)]
        )

    def forward(self, embeddings: torch.Tensor, slot_mask: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [B, num_slots, slices_per_slot, embed_dim]
        slot_mask:  [B, num_slots]   (1 = present, 0 = absent)
        returns:    [B, num_targets] raw logits (feed to BCEWithLogitsLoss)
        """
        B = embeddings.shape[0]
        T = self.num_slots * self.slices_per_slot

        tokens = embeddings.view(B, T, -1)              # [B, T, embed_dim]
        tokens = self.proj(tokens)                        # [B, T, proj_dim]

        # Expand the per-slot mask to per-token, then to an additive bias.
        token_mask = slot_mask.unsqueeze(-1).expand(B, self.num_slots, self.slices_per_slot)
        token_mask = token_mask.reshape(B, T)              # [B, T]
        additive_mask = (1.0 - token_mask) * -1e4          # 0 where present, -1e4 where absent

        scores = self.attn_scorer(tokens)                  # [B, T, num_targets]
        scores = scores + additive_mask.unsqueeze(-1)       # broadcast mask over targets
        weights = torch.softmax(scores, dim=1)               # softmax over T, per target

        # Weighted pool per disease: [B, num_targets, proj_dim]
        pooled = torch.einsum("btn,btd->bnd", weights, tokens)

        logits = torch.cat(
            [self.classifiers[i](pooled[:, i, :]) for i in range(self.num_targets)],
            dim=1,
        )  # [B, num_targets]
        return logits


# ============================================================================
# STEP 4b (was dataset.py) — PyTorch Dataset with embedding caching
# Restricted to the 58 labeled studies: get_or_build_embedding() is only
# ever called with a study_uid that came from the filtered gold_df in
# main(), so DICOM access never touches any other study folder.
# ============================================================================
CACHE_DIR = os.path.join(OUTPUT_DIR, "embed_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

_train_meta_lookup = None   # lazy-loaded, shared across calls


def _get_train_meta_lookup() -> dict:
    global _train_meta_lookup
    if _train_meta_lookup is None:
        _train_meta_lookup = _load_series_meta_lookup(TRAIN_SERIES_META_CSV)
    return _train_meta_lookup


def get_or_build_embedding(study_uid: str, dicom_root: str, meta_lookup: dict):
    cache_path = os.path.join(CACHE_DIR, f"{study_uid}.pt")
    if os.path.exists(cache_path):
        obj = torch.load(cache_path, weights_only=False)
        return obj["embeddings"], obj["slot_mask"]

    study_dir = os.path.join(dicom_root, str(study_uid))
    metas = scan_study(study_uid, study_dir, meta_lookup)
    slot_slices = preprocess_study(metas)
    embeddings, slot_mask = embed_study(slot_slices)
    torch.save({"embeddings": embeddings, "slot_mask": slot_mask}, cache_path)
    return embeddings, slot_mask


class RSNAKneeDataset(Dataset):
    """labels_df needs a StudyInstanceUID column plus one column per TARGETS.
    is_real_only: if True only loads real-labeled rows (for validation).
    sample_weight: real=1.0, weak=0.3 (downweights noisy parser labels)."""

    def __init__(self, labels_df: pd.DataFrame,
                 is_weak: bool = False, is_real_only: bool = False):
        df = labels_df.reset_index(drop=True)
        if is_real_only and "is_real_label" in df.columns:
            df = df[df["is_real_label"].astype(bool)].reset_index(drop=True)
        self.df      = df
        self.is_weak = is_weak

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        embeddings, slot_mask = get_or_build_embedding(
            study_uid, TRAIN_DICOM_DIR, _get_train_meta_lookup()
        )
        labels = torch.tensor(row[TARGETS].to_numpy().astype("float32"))
        is_real = bool(row.get("is_real_label", True))
        sample_weight = torch.tensor(1.0 if is_real else 0.3)
        return embeddings, slot_mask, labels, sample_weight, study_uid


# ============================================================================
# STEP 5 (was train.py) — 5-fold GroupKFold training on the 58 labeled
# studies only. The original weak-label mixing branch is disabled
# (USE_WEAK_LABELS=False above).
# ============================================================================
def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=min(NUM_CPU_THREADS, 4),
        pin_memory=(DEVICE == "cuda"),
    )


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    all_logits, all_labels = [], []
    total_loss = 0.0

    for embeddings, slot_mask, labels, sample_weight, _ in loader:
        embeddings, slot_mask = embeddings.to(DEVICE), slot_mask.to(DEVICE)
        labels, sample_weight = labels.to(DEVICE), sample_weight.to(DEVICE)

        with torch.set_grad_enabled(train):
            logits = model(embeddings, slot_mask)
            loss = criterion(logits, labels)                   # [B, num_targets]
            loss = (loss.mean(dim=1) * sample_weight).mean()     # weight weak-labeled rows down

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * embeddings.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    return total_loss / len(loader.dataset), logits, labels


def mean_auc(logits, labels) -> float:
    probs = 1 / (1 + np.exp(-logits))
    aucs = []
    for i in range(NUM_TARGETS):
        if len(np.unique(labels[:, i])) < 2:
            continue  # skip a fold/target with only one class present
        aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def load_gold_df() -> pd.DataFrame:
    """Load training labels.
    If USE_WEAK_LABELS=True and PARSED_LABELS_CSV exists:
      returns 58 real + (N_TOTAL_STUDIES-58) weak-labeled studies.
      is_real_label column marks which are official ground truth.
    Otherwise: returns only the 58 officially labeled studies."""
    train = pd.read_csv(TRAIN_CSV, dtype={"StudyInstanceUID": str})
    has_labels = train[TARGETS].notna().all(axis=1)
    gold_df = train.loc[has_labels, ["StudyInstanceUID"] + TARGETS].reset_index(drop=True)
    for t in TARGETS:
        gold_df[t] = gold_df[t].astype("float32")
    gold_df["is_real_label"] = True
    print(f"[train] Official labeled studies: {len(gold_df)}")

    if USE_WEAK_LABELS and PARSED_LABELS_CSV and os.path.isfile(PARSED_LABELS_CSV):
        parsed = pd.read_csv(PARSED_LABELS_CSV, dtype={"StudyInstanceUID": str})
        parsed[TARGETS] = parsed[TARGETS].fillna(0).astype("float32")
        if "is_real_label" not in parsed.columns:
            parsed["is_real_label"] = False
        # exclude real-labeled studies from weak pool (no overlap)
        real_uids = set(gold_df["StudyInstanceUID"])
        weak_pool = parsed[~parsed["StudyInstanceUID"].isin(real_uids)].copy()
        n_weak = max(0, N_TOTAL_STUDIES - len(gold_df))
        n_weak = min(n_weak, len(weak_pool))
        weak_sample = weak_pool.sample(n=n_weak, random_state=42) if n_weak > 0 else weak_pool.iloc[:0]
        weak_sample = weak_sample[["StudyInstanceUID"] + TARGETS + ["is_real_label"]].copy()
        combined = pd.concat([gold_df, weak_sample], ignore_index=True)
        print(f"[train] Total training pool: {len(combined)} "
              f"({len(gold_df)} real + {len(weak_sample)} weak)")
        return combined

    print(f"[train] Weak labels not available — training on {len(gold_df)} official studies only")
    return gold_df


def train_all_folds():
    gold_df  = load_gold_df()

    # checkpoint/OOF resume: if all fold files + oof exist, skip training
    _oof_path   = os.path.join(OUTPUT_DIR, "oof_predictions.csv")
    _all_ckpts  = all(os.path.isfile(os.path.join(OUTPUT_DIR, f"fold{f}_best.pt"))
                      for f in range(N_FOLDS))
    if _all_ckpts and os.path.isfile(_oof_path):
        print("[train] All fold checkpoints + OOF found — skipping training entirely")
        _oof = pd.read_csv(_oof_path)
        print(f"[train] OOF loaded: {_oof.shape}")
        return

    # validate-only studies = real-labeled ones (is_real_label=True)
    real_mask = gold_df.get("is_real_label", pd.Series([True]*len(gold_df))).astype(bool)
    val_pool  = gold_df[real_mask].reset_index(drop=True)

    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = val_pool["StudyInstanceUID"]
    fold_scores = []
    oof_preds   = np.full((len(val_pool), NUM_TARGETS), np.nan, dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(gkf.split(val_pool, groups=groups)):
        fold_path = os.path.join(OUTPUT_DIR, f"fold{fold}_best.pt")
        print(f"\n=== Fold {fold} ===")
        # val = real-labeled studies in this fold
        val_gold   = val_pool.iloc[val_idx]
        # train = ALL studies (real + weak) minus val
        val_uids   = set(val_gold["StudyInstanceUID"])
        train_gold = gold_df[~gold_df["StudyInstanceUID"].isin(val_uids)].copy()

        train_ds = RSNAKneeDataset(train_gold)
        val_ds   = RSNAKneeDataset(val_gold, is_real_only=True)

        train_loader = make_loader(train_ds, shuffle=True)
        val_loader   = make_loader(val_ds,   shuffle=False)

        if os.path.isfile(fold_path):
            print(f"  checkpoint exists — loading {fold_path}, skipping training")
            model = DiseaseAttentionModel().to(DEVICE)
            model.load_state_dict(torch.load(fold_path, map_location=DEVICE, weights_only=False))
        else:
            model     = DiseaseAttentionModel().to(DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            criterion = nn.BCEWithLogitsLoss(reduction="none")
            best_auc  = -1.0
            for epoch in range(EPOCHS):
                train_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer)
                val_loss, val_logits, val_labels = run_epoch(model, val_loader, criterion)
                val_auc = mean_auc(val_logits, val_labels)
                print(f"  epoch {epoch:02d}  train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  val_auc={val_auc:.4f}")
                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), fold_path)
            # reload best checkpoint
            model.load_state_dict(torch.load(fold_path, map_location=DEVICE, weights_only=False))
            fold_scores.append(best_auc)
            print(f"  fold {fold} best OOF AUC: {best_auc:.4f}")

        # OOF predictions for val studies
        model.eval()
        with torch.no_grad():
            _, val_logits, _ = run_epoch(model, val_loader, criterion=nn.BCEWithLogitsLoss(reduction="none"))
        probs = 1 / (1 + np.exp(-val_logits))
        oof_preds[val_idx] = probs

    # save OOF
    oof_df = pd.DataFrame(oof_preds, columns=TARGETS)
    oof_df.insert(0, "StudyInstanceUID", val_pool["StudyInstanceUID"].values)
    oof_df.to_csv(_oof_path, index=False)
    if fold_scores:
        print(f"\nMean OOF AUC across {N_FOLDS} folds: {np.mean(fold_scores):.4f}")
    print(f"OOF saved: {_oof_path}")


# ============================================================================
# STEP 6 (was kaggle_pipeline.py) — inference on the test set
# ============================================================================
def load_fold_models() -> list[DiseaseAttentionModel]:
    models = []
    for fold in range(N_FOLDS):
        ckpt_path = os.path.join(OUTPUT_DIR, f"fold{fold}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"[predict] WARNING: missing {ckpt_path}, skipping fold {fold}")
            continue
        model = DiseaseAttentionModel().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        model.eval()
        models.append(model)
    if not models:
        raise RuntimeError("No fold checkpoints found — run training first.")
    return models


@torch.no_grad()
def predict_study(models: list[DiseaseAttentionModel], study_uid: str,
                   test_meta_lookup: dict) -> np.ndarray:
    study_dir = os.path.join(TEST_DICOM_DIR, str(study_uid))
    metas = scan_study(study_uid, study_dir, test_meta_lookup)
    slot_slices = preprocess_study(metas)
    embeddings, slot_mask = embed_study(slot_slices)

    embeddings = embeddings.unsqueeze(0).to(DEVICE)   # add batch dim
    slot_mask = slot_mask.unsqueeze(0).to(DEVICE)

    fold_probs = []
    for model in models:
        logits = model(embeddings, slot_mask)
        fold_probs.append(torch.sigmoid(logits).cpu().numpy()[0])

    return np.mean(fold_probs, axis=0)   # average across folds


def _load_test_study_ids() -> pd.DataFrame:
    """Prefer sample_submission.csv (gives canonical column order too);
    fall back to test.csv (StudyInstanceUID only) if that's missing."""
    if os.path.isfile(SAMPLE_SUBMISSION_CSV):
        return pd.read_csv(SAMPLE_SUBMISSION_CSV, dtype={"StudyInstanceUID": str})
    if os.path.isfile(TEST_CSV):
        df = pd.read_csv(TEST_CSV, dtype={"StudyInstanceUID": str})
        for t in TARGETS:
            df[t] = 0.5
        return df
    raise FileNotFoundError(f"Neither {SAMPLE_SUBMISSION_CSV} nor {TEST_CSV} found")


def predict_all():
    sample_sub = _load_test_study_ids()
    study_ids = sample_sub["StudyInstanceUID"].tolist()
    test_meta_lookup = _load_series_meta_lookup(TEST_SERIES_META_CSV)

    models = load_fold_models()

    rows = []
    for i, study_uid in enumerate(study_ids):
        probs = predict_study(models, study_uid, test_meta_lookup)
        rows.append([study_uid, *probs])
        if i % 20 == 0:
            print(f"[predict] {i}/{len(study_ids)} studies done")

    submission = pd.DataFrame(rows, columns=["StudyInstanceUID", *TARGETS])
    out_path = os.path.join(OUTPUT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"[predict] wrote {out_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--predict-only", action="store_true")
    args = ap.parse_args()

    # ── KAGGLE CACHE RESTORE ─────────────────────────────────────────────
    if ON_KAGGLE:
        _cache_root = "/kaggle/input/datasets/kabirverma01/rsna-v4-xgboost-cache/rsnav4_ssh"
        if os.path.isdir(_cache_root):
            import shutil as _shutil
            print(f"  Restoring cache from {_cache_root} ...")
            for _sub in ("embed_cache", "models"):
                _src = os.path.join(_cache_root, _sub)
                if os.path.isdir(_src):
                    _dst = os.path.join(OUTPUT_DIR, _sub)
                    _shutil.copytree(_src, _dst, dirs_exist_ok=True)
                    _n = sum(1 for _ in os.scandir(_dst))
                    print(f"    restored {_sub}/ ({_n} items)")
            # also restore fold checkpoints sitting at cache root
            for _f in os.listdir(_cache_root):
                if _f.startswith("fold") and _f.endswith(".pt"):
                    _shutil.copy2(os.path.join(_cache_root, _f),
                                  os.path.join(OUTPUT_DIR, _f))
                    print(f"    restored {_f}")
            if os.path.isfile(os.path.join(_cache_root, "oof_predictions.csv")):
                _shutil.copy2(os.path.join(_cache_root, "oof_predictions.csv"),
                              os.path.join(OUTPUT_DIR, "oof_predictions.csv"))
                print("    restored oof_predictions.csv")
            print("  Cache restore done.")
        else:
            print("  No cache dataset attached — starting fresh.")

    print("=" * 70)
    print("RSNA KNEE — merged local pipeline")
    print(f"  On Kaggle       : {ON_KAGGLE}")
    print(f"  Device          : {DEVICE}")
    print(f"  Data dir        : {DATA_DIR}")
    print(f"  MedSigLIP path  : {MEDSIGLIP_PATH}")
    print(f"  Output dir      : {OUTPUT_DIR}")
    print(f"  Weak labels     : {USE_WEAK_LABELS}  (restricted to 58 labeled studies)")
    print("=" * 70)

    if not args.predict_only:
        train_all_folds()
    if not args.train_only:
        predict_all()


if __name__ == "__main__":
    main()
