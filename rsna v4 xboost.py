# ============================================================
# RSNA KNEE ABNORMALITY DETECTION
# Full pipeline: DICOM → MedSigLIP → Attention Model → Submission
#
# Kaggle paths:
#   Data    : /kaggle/input/rsna-knee-abnormality-2024/
#   MedSigLIP: /kaggle/input/medsiglip/
#   Output  : /kaggle/working/
# ============================================================

import os
import re
import math
import random
import pickle
import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA

# ── config ────────────────────────────────────────────────────────────────────
IS_KAGGLE     = os.path.exists("/kaggle/input")

# Linux SSH/GPU server (not Kaggle, not the local Windows dev machine).
# Same server layout used by RSNA v4.
IS_SERVER     = (not IS_KAGGLE) and os.name == "posix"
SERVER_ROOT   = Path("/home/harleen_ece/rsna_knee_ai")

def _find_data_root():
    # Handles both the raw competition dataset and a manually-renamed copy.
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-2024"),
    ]
    for c in candidates:
        if (c / "train_series.csv").exists() or (c / "train.csv").exists():
            print(f"  Data root found at: {c}")
            return c
    print("  Scanning /kaggle/input (max depth 3) for competition data (train_series.csv)...")
    # bounded-depth scan instead of unbounded rglob — avoids crawling into
    # the (huge) train_series/<study>/<series>/*.dcm tree, which is what
    # made the old unbounded rglob effectively hang for minutes.
    base = Path("/kaggle/input")
    for depth in range(0, 4):
        pattern = "/".join(["*"] * depth + ["train_series.csv"]) if depth else "train_series.csv"
        for p in sorted(base.glob(pattern)):
            print(f"  Data root found at: {p.parent}")
            return p.parent
    raise RuntimeError(
        "Competition data not found under /kaggle/input.\n"
        "Attach the RSNA Knee Abnormality Detection competition data to this notebook."
    )

if IS_KAGGLE:
    DATA_ROOT = _find_data_root()
elif IS_SERVER:
    DATA_ROOT = Path("/home/harleen_ece/rsna_knee_ai/DATA")
else:
    DATA_ROOT = Path("C:/kabir/RSNA_Knee_AI/DATA")
# auto-find MedSigLIP on Kaggle — handles subfolder variations
def _find_medsiglip():
    # kabirverma01/medsiglip dataset — files are in root
    candidates = [
        Path("/kaggle/input/datasets/kabirverma01/medsiglip/MedSigLIP"),  # confirmed actual path
        Path("/kaggle/input/datasets/kabirverma01/medsiglip"),
        Path("/kaggle/input/medsiglip"),           # dataset slug = medsiglip
        Path("/kaggle/input/medsiglip/MedSigLIP"), # if Kaggle adds subfolder
    ]
    for c in candidates:
        if (c / "config.json").exists():
            print(f"  MedSigLIP found at: {c}")
            return c
    # fallback: bounded-depth scan of /kaggle/input (avoids crawling into
    # the huge train_series/<study>/<series>/*.dcm tree)
    print("  Scanning /kaggle/input (max depth 3) for MedSigLIP...")
    base = Path("/kaggle/input")
    for depth in range(0, 4):
        pattern = "/".join(["*"] * depth + ["config.json"]) if depth else "config.json"
        for p in sorted(base.glob(pattern)):
            if (p.parent / "model.safetensors").exists():
                print(f"  MedSigLIP found at: {p.parent}")
                return p.parent
    raise RuntimeError(
        "MedSigLIP not found.\n"
        "Attach dataset kabirverma01/medsiglip to this notebook."
    )

if IS_KAGGLE:
    MODEL_PATH = _find_medsiglip()
elif IS_SERVER:
    MODEL_PATH = Path("/home/harleen_ece/rsna_knee_ai/MedSigLIP")
else:
    MODEL_PATH = Path("C:/kabir/RSNA_Knee_AI/MedSigLIP")

if IS_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
elif IS_SERVER:
    WORK_DIR = SERVER_ROOT / "kaggle_run"
else:
    WORK_DIR = Path("C:/kabir/RSNA/kaggle_run")
TRAIN_SERIES  = DATA_ROOT / "train_series"
TEST_SERIES   = DATA_ROOT / "test_series"
EMB_DIR       = WORK_DIR / "embeddings"
MODEL_DIR     = WORK_DIR / "models"
WORK_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _kaggle_dataset_variants(slug):
    """
    Kaggle sometimes mounts an attached dataset at /kaggle/input/<slug>
    and sometimes nests it under /kaggle/input/datasets/<username>/<slug>
    (confirmed for this account: MedSigLIP landed at
    /kaggle/input/datasets/kabirverma01/medsiglip). Returns both shapes so
    callers can just check each path.
    """
    base = Path("/kaggle/input")
    variants = [base / slug]
    datasets_dir = base / "datasets"
    if datasets_dir.exists():
        for user_dir in datasets_dir.iterdir():
            if user_dir.is_dir():
                variants.append(user_dir / slug)
    return variants

# If a previous Kaggle run's WORK_DIR (embeddings, models, index CSVs) was
# saved as its own Dataset and re-attached, copy everything back into place
# up front. embed_slots() and the Stage A / Stage B resume checks each skip
# recomputation when their target files already exist, so this picks up
# exactly where a previous run left off instead of starting over.
if IS_KAGGLE:
    _resume_dataset_slugs = ["rsnav3-trained", "rsna-embeddings-cache"]
    _resume_root = None
    for _slug in _resume_dataset_slugs:
        for v in _kaggle_dataset_variants(_slug):
            if v.exists() and (any(v.rglob("*.pt")) or (v / "embeddings").exists()):
                _resume_root = v
                break
        if _resume_root:
            break

    if _resume_root:
        import shutil as _shutil
        print(f"  Found previous run output at {_resume_root}, restoring into {WORK_DIR} ...")

        # embeddings/ and models/ subfolders — copy whole trees
        for _sub in ("embeddings", "models"):
            _src = _resume_root / _sub
            if _src.exists():
                _dst = WORK_DIR / _sub
                _shutil.copytree(_src, _dst, dirs_exist_ok=True)
                print(f"    restored {_sub}/  ({sum(1 for _ in _dst.rglob('*') if _.is_file())} files)")

        # the two index CSVs sit at the dataset root
        for _idx_name in ("train_dicom_index.csv", "train_embedding_index.csv"):
            _idx_src = _resume_root / _idx_name
            if _idx_src.exists():
                _shutil.copy2(_idx_src, WORK_DIR / _idx_name)
                print(f"    restored {_idx_name}")
    else:
        print("  No previous-run cache dataset found — starting fresh.")

# Big zip holding the (mostly-unlabeled) train series that are NOT already
# unzipped on disk. Only used LOCALLY — on Kaggle, data is already unzipped
# (mounted read-only from the attached Dataset), so this path is unused there.
TRAIN_SERIES_ZIP = Path(r"D:\rsna-knee-abnormality-detection.zip") if not IS_KAGGLE else None

# Parser output (weak labels for everyone + real labels for 58) produced by
# train_and_predict.py / report_parser.py. This REPLACES train.csv as the
# label source. On Kaggle, upload this CSV as its own small Dataset (or
# alongside train_series.csv in the main data dataset) and attach it —
# code auto-finds it under /kaggle/input if the exact path below isn't found.
if IS_KAGGLE:
    _parsed_candidates = [v / "final_labels_real_plus_generated.csv"
                          for v in _kaggle_dataset_variants("final-labels-real-plus-generated")]
    _parsed_candidates += [v / "final_labels_real_plus_generated.csv"
                           for v in _kaggle_dataset_variants("rsna-knee-labels")]
    _parsed_candidates.append(DATA_ROOT / "final_labels_real_plus_generated.csv")
    PARSED_LABELS_CSV = next((p for p in _parsed_candidates if p.exists()), _parsed_candidates[0])
elif IS_SERVER:
    PARSED_LABELS_CSV = Path(
        "/home/harleen_ece/rsna_knee_ai/AI-MODEL/final_labels_real_plus_generated.csv"
    )
else:
    PARSED_LABELS_CSV = Path(r"C:\kabir\RSNA_Knee_AI\parser\output\final_labels_real_plus_generated.csv")

N_TOTAL_STUDIES = 4340  # 58 real-labeled + (N_TOTAL_STUDIES - 58) weak-labeled
SAMPLE_SEED     = 42

# ── constants ─────────────────────────────────────────────────────────────────
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

SLOT_NAMES = [
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
]
N_SLOT    = len(SLOT_NAMES)
EMBED_DIM = 1152
PROJ_DIM  = 256
MAX_SLICES  = 12
BATCH_SIZE  = 16  # T4 has 16GB VRAM, safe at 16
SLICE_BAND  = (0.20, 0.80)
LAT_OFFSET  = 20.0
PRIOR_STRENGTH = 0.55

SLOT_PRIOR = {
    "ACL":              [1, 0, 0, 1, 0, 1],
    "MCL":              [0, 1, 0, 0, 1, 0],
    "Medial Meniscus":  [1, 1, 0, 1, 1, 0],
    "Lateral Meniscus": [1, 1, 0, 1, 1, 0],
    "Medial OA":        [0, 1, 0, 0, 1, 0],
    "Lateral OA":       [0, 1, 0, 0, 1, 0],
    "PF OA":            [1, 0, 1, 0, 0, 1],
    "Effusion":         [1, 0, 1, 0, 0, 0],
    "Synovitis":        [1, 0, 1, 0, 0, 0],
    "Baker's":          [1, 0, 0, 0, 0, 0],
    "Contusion":        [1, 1, 0, 1, 1, 0],
    "Fracture":         [1, 1, 1, 1, 1, 1],
}

SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", 1, 1),
    ("COR_FLUID_FS",   "Coronal",  1, 1),
    ("AX_FLUID_FS",    "Axial",    1, 1),
    ("SAG_FLUID_NOFS", "Sagittal", 1, 0),
    ("COR_T1",         "Coronal",  0, 0),
    ("SAG_T1",         "Sagittal", 0, 0),
]


def seed_everything(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — SELECTIVE UNZIP (only for studies we actually need, only once)
# ══════════════════════════════════════════════════════════════════════════════
def ensure_studies_unzipped(study_ids, series_root, zip_path):
    """
    Make sure `series_root` contains a folder for every StudyInstanceUID in
    `study_ids`. Studies whose folder already exists on disk are left
    untouched (no re-extraction). Anything missing is pulled out of the big
    `zip_path` archive, and only that study's files are extracted — nothing
    else. Safe to call every run: on a second run everything is already
    there so it just does a fast existence check and returns.
    """
    import zipfile

    series_root = Path(series_root)
    series_root.mkdir(parents=True, exist_ok=True)

    study_ids = set(str(s) for s in study_ids)
    already = {p.name for p in series_root.iterdir() if p.is_dir()}
    missing = study_ids - already

    if not missing:
        print(f"  All {len(study_ids)} requested studies already on disk — nothing to unzip.")
        return

    if not Path(zip_path).exists():
        print(f"  [WARN] {len(missing)} studies missing and zip not found at {zip_path} — skipping unzip.")
        return

    print(f"  {len(already)} studies already on disk, {len(missing)} need extracting from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Archive layout is train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
        # (a "train_series" folder at the zip root, not study folders at root).
        # Find that prefix once, then pull out only the members whose study
        # folder is one we're missing.
        names = zf.namelist()
        prefix = ""
        for n in names:
            parts = n.split("/")
            if len(parts) > 1 and parts[0].lower() == "train_series":
                prefix = "train_series/"
                break

        wanted_members = [n for n in names
                          if n[len(prefix):].split("/")[0] in missing]
        print(f"  Extracting {len(wanted_members)} files for {len(missing)} studies "
              f"(archive prefix: '{prefix}')...")
        for member in tqdm(wanted_members, desc="  Unzipping"):
            # extract, then relocate out from under the "train_series/" prefix
            # so the result lands directly as series_root/<StudyInstanceUID>/...
            zf.extract(member, path=series_root)
            if prefix:
                extracted_path = series_root / member
                rel = Path(member[len(prefix):])
                target_path = series_root / rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if extracted_path != target_path:
                    extracted_path.replace(target_path)

        # clean up the now-empty "train_series" subfolder left behind by extraction
        if prefix:
            leftover = series_root / "train_series"
            if leftover.exists():
                for root, dirs, files in os.walk(leftover, topdown=False):
                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except OSError:
                            pass
                try:
                    os.rmdir(leftover)
                except OSError:
                    pass

    still_missing = missing - {p.name for p in series_root.iterdir() if p.is_dir()}
    if still_missing:
        print(f"  [WARN] {len(still_missing)} studies not found in zip either: "
              f"{sorted(still_missing)[:5]}{'...' if len(still_missing) > 5 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCAN DICOMS
# ══════════════════════════════════════════════════════════════════════════════
def scan_dicoms(series_root, series_csv, study_filter=None):
    """
    Scan DCMs, join with series CSV for slot metadata.
    study_filter: optional iterable of StudyInstanceUIDs to restrict the
    scan to — goes straight to each study's folder instead of walking the
    entire series_root tree, which matters a lot when series_root holds
    far more studies than we're actually training on.
    """
    series_root = Path(series_root)
    if study_filter is not None:
        study_filter = list(study_filter)
        paths = []
        for uid in study_filter:
            study_dir = series_root / str(uid)
            if study_dir.is_dir():
                paths.extend(study_dir.rglob("*.dcm"))
        print(f"  Restricting scan to {len(study_filter)} study folders → {len(paths)} DCMs")
    else:
        paths = list(series_root.rglob("*.dcm"))
    print(f"  DCM files found: {len(paths)}")

    def _read_header(p):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            study  = str(getattr(ds, "StudyInstanceUID",  "") or "").strip()
            series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
            sop    = str(getattr(ds, "SOPInstanceUID",    "") or "").strip()
            inst   = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                try: inst = int(inst)
                except: inst = None
            if study and series:
                return dict(filepath=str(p), StudyInstanceUID=study,
                            SeriesInstanceUID=series, SOPInstanceUID=sop,
                            InstanceNumber=inst)
        except Exception:
            pass
        return None

    from concurrent.futures import ThreadPoolExecutor
    print(f"  Reading {len(paths)} DICOM headers (parallel)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_read_header, paths))
    rows = [r for r in results if r is not None]

    df = pd.DataFrame(rows)
    meta = pd.read_csv(series_csv, dtype=str)
    meta["Fluid_Sensitive"] = meta["Fluid_Sensitive"].astype(int)
    meta["Fat_Suppression"] = meta["Fat_Suppression"].astype(int)

    merged = df.merge(meta, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

    # fallback: infer plane from IOP for unmatched
    unmatched = merged["Anatomical_Plane"].isna()
    if unmatched.sum() > 0:
        for idx in merged[unmatched].index:
            try:
                ds = pydicom.dcmread(merged.loc[idx, "filepath"],
                                     stop_before_pixels=True, force=True)
                merged.loc[idx, "Anatomical_Plane"] = _plane_from_iop(ds)
                merged.loc[idx, "Fluid_Sensitive"]  = 0
                merged.loc[idx, "Fat_Suppression"]  = 0
            except Exception:
                pass

    print(f"  Studies: {merged['StudyInstanceUID'].nunique()}")
    print(f"  Series : {merged['SeriesInstanceUID'].nunique()}")
    return merged


def _plane_from_iop(ds):
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        r, c = iop[:3], iop[3:]
        n = [r[1]*c[2]-r[2]*c[1], r[2]*c[0]-r[0]*c[2], r[0]*c[1]-r[1]*c[0]]
        a = [abs(x) for x in n]
        if a[0] > a[1] and a[0] > a[2]: return "Sagittal"
        if a[1] > a[0] and a[1] > a[2]: return "Coronal"
        if a[2] > a[0] and a[2] > a[1]: return "Axial"
    except Exception:
        pass
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD SLOTS + LATERALITY
# ══════════════════════════════════════════════════════════════════════════════
def detect_laterality(dicom_df):
    study_cx = {}
    for _, r in dicom_df.iterrows():
        try:
            ds  = pydicom.dcmread(r["filepath"], stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ps  = getattr(ds, "PixelSpacing", None)
            rows = getattr(ds, "Rows", None)
            cols = getattr(ds, "Columns", None)
            if ipp is None or iop is None or ps is None: continue
            ipp = np.array([float(x) for x in ipp[:3]])
            iop = np.array([float(x) for x in iop[:6]])
            ps  = np.array([float(x) for x in ps[:2]])
            cx  = ipp[0] + iop[0]*ps[1]*float(cols)/2 + iop[3]*ps[0]*float(rows)/2
            study_cx.setdefault(r["StudyInstanceUID"], []).append(float(cx))
        except Exception:
            pass
    result = {}
    for su, xs in study_cx.items():
        m = float(np.median(xs))
        result[su] = "R" if m < -LAT_OFFSET else ("L" if m > LAT_OFFSET else None)
    return result


def assign_slots(dicom_df):
    slice_counts = dicom_df.groupby("SeriesInstanceUID").size().to_dict()
    series_df = (dicom_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])
                 .first().reset_index())
    series_df["n_slices"] = series_df["SeriesInstanceUID"].map(slice_counts)
    rows = []
    for study, grp in series_df.groupby("StudyInstanceUID"):
        for slot_name, plane, fluid, fat in SLOTS:
            mask = ((grp["Anatomical_Plane"] == plane) &
                    (grp["Fluid_Sensitive"].fillna(0).astype(int) == fluid) &
                    (grp["Fat_Suppression"].fillna(0).astype(int) == fat))
            cands = grp[mask].sort_values("n_slices", ascending=False)
            if len(cands) == 0:
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": "",
                             "slot_name": slot_name, "n_slices": 0, "presence_mask": 0})
            else:
                best = cands.iloc[0]
                rows.append({"StudyInstanceUID": study,
                             "SeriesInstanceUID": best["SeriesInstanceUID"],
                             "slot_name": slot_name,
                             "n_slices": int(best["n_slices"]),
                             "presence_mask": 1})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MEDSIGLIP EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
def sort_slices(paths):
    meta = []
    for p in paths:
        try:
            ds   = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            ipp  = getattr(ds, "ImagePositionPatient", None)
            inst = getattr(ds, "InstanceNumber", None)
            pos  = None
            if ipp is not None:
                coords = np.array([float(x) for x in ipp[:3]])
                if np.isfinite(coords).all(): pos = coords
            meta.append((p, pos, inst))
        except Exception:
            meta.append((p, None, None))

    positioned = [(p, pos, inst) for p, pos, inst in meta if pos is not None]
    if len(positioned) >= max(2, int(0.8 * len(meta))):
        xyz  = np.stack([pos for _, pos, _ in positioned])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        meta.sort(key=lambda x: (float(x[1][axis]) if x[1] is not None else spare,
                                  float(x[2]) if x[2] is not None else float("inf")))
    else:
        meta.sort(key=lambda x: (float(x[2]) if x[2] is not None else float("inf"),))
    return [p for p, _, _ in meta]


def select_band(paths):
    n  = len(paths)
    lo = int(np.floor(n * SLICE_BAND[0]))
    hi = int(np.ceil(n  * SLICE_BAND[1]))
    band = paths[lo:hi]
    if len(band) <= MAX_SLICES: return band
    idx = np.unique(np.round(np.linspace(0, len(band)-1, MAX_SLICES)).astype(int))
    return [band[i] for i in idx]


def normalise_laterality(imgs, plane, lat):
    if lat != "R": return imgs
    if plane in ("Coronal", "Axial"):
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
    return imgs[::-1]


def dicom_to_pil(path):
    ds  = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo: lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo: arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def load_medsiglip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading MedSigLIP | device={device}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    return processor, model, device


@torch.inference_mode()
def encode_images(images, processor, model, device):
    feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch  = images[i:i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        if device == "cuda": pixels = pixels.to(dtype=torch.float16)
        out = model.get_image_features(pixel_values=pixels)
        if not torch.is_tensor(out):
            if hasattr(out, "pooler_output"):       out = out.pooler_output
            elif hasattr(out, "image_embeds"):      out = out.image_embeds
            elif hasattr(out, "last_hidden_state"): out = out.last_hidden_state.mean(1)
        out = F.normalize(out.float(), dim=-1)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)


def embed_slots(slots_df, dicom_df, processor, model, device,
                lat_map, out_dir, force=False):
    series_to_files = (dicom_df.groupby("SeriesInstanceUID")["filepath"]
                       .apply(list).to_dict())
    present = slots_df[slots_df["presence_mask"] == 1].copy()
    index_rows = []
    done = failed = skipped = 0

    for _, row in tqdm(present.iterrows(), total=len(present), desc="  Embedding"):
        study  = str(row["StudyInstanceUID"])
        series = str(row["SeriesInstanceUID"])
        slot   = str(row["slot_name"])
        plane  = str(row.get("Anatomical_Plane", "Unknown"))
        lat    = lat_map.get(study)

        out_path = out_dir / study / f"{series}__{slot}.pt"
        if out_path.exists() and not force:
            skipped += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                "slot_name": slot, "embedding_file": str(out_path),
                                "presence_mask": 1})
            continue

        paths = [Path(p) for p in series_to_files.get(series, []) if Path(p).is_file()]
        paths = sort_slices(paths)
        paths = select_band(paths)

        images = []
        for p in paths:
            try: images.append(dicom_to_pil(p))
            except Exception: pass

        if not images:
            failed += 1
            continue

        images = normalise_laterality(images, plane, lat)

        try:
            feats = encode_images(images, processor, model, device)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": feats, "slot_name": slot,
                        "study_uid": study, "series_uid": series,
                        "laterality": lat, "plane": plane,
                        "n_slices": len(images)}, out_path)
            done += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                "slot_name": slot, "embedding_file": str(out_path),
                                "presence_mask": 1})
        except Exception as e:
            print(f"\n  [WARN] {study[:20]}/{slot}: {e}")
            failed += 1

    print(f"  Embedded={done} Skipped={skipped} Failed={failed}")
    return pd.DataFrame(index_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
class SlotAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.att   = nn.Linear(PROJ_DIM, len(TARGETS), bias=False)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(PROJ_DIM), nn.Linear(PROJ_DIM, 64),
                          nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))
            for _ in TARGETS
        ])
        prior = torch.zeros(len(TARGETS), N_SLOT)
        for t, target in enumerate(TARGETS):
            for s, val in enumerate(SLOT_PRIOR[target]):
                prior[t, s] = val * PRIOR_STRENGTH
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask, slot_indices):
        h       = self.proj(x)
        scores  = self.att(h).T
        scores  = scores + self.slot_prior[:, slot_indices]
        absent  = (mask[slot_indices] < 0.5)
        scores  = scores.masked_fill(absent.unsqueeze(0), -1e4)
        weights = torch.softmax(scores, dim=1)
        outputs = []
        for t in range(len(TARGETS)):
            pooled = (weights[t, :, None] * h).sum(dim=0)
            outputs.append(self.heads[t](pooled).squeeze())
        return torch.stack(outputs)


def load_embedding(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "embeddings" in obj: x = obj["embeddings"]
    elif torch.is_tensor(obj): x = obj
    else:
        cands = [v for v in (obj.values() if isinstance(obj, dict) else [])
                 if torch.is_tensor(v)]
        if not cands: raise ValueError(f"No tensor in {path}")
        x = cands[0]
    x = x.float()
    if x.ndim == 1: x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != EMBED_DIM:
        raise ValueError(f"{path}: expected [N,{EMBED_DIM}]")
    return x


class StudyDataset:
    def __init__(self, emb_df, labels_df):
        self.emb_df = emb_df
        self.labels = labels_df.set_index("StudyInstanceUID")
        self.ids    = sorted(emb_df["StudyInstanceUID"].unique())

    def __len__(self): return len(self.ids)

    def get(self, i):
        study = self.ids[i]
        rows  = self.emb_df[self.emb_df["StudyInstanceUID"] == study]
        slot_to_file = {r["slot_name"]: r["embedding_file"]
                        for _, r in rows.iterrows() if r["presence_mask"] == 1}
        tensors, slot_indices, mask = [], [], torch.zeros(N_SLOT)
        for s_idx, slot_name in enumerate(SLOT_NAMES):
            if slot_name in slot_to_file:
                try:
                    x = load_embedding(slot_to_file[slot_name])
                    tensors.append(x)
                    slot_indices.extend([s_idx] * len(x))
                    mask[s_idx] = 1.0
                except Exception as e:
                    print(f"  [WARN] {study[:20]}/{slot_name}: {e}")
        if not tensors:
            tensors = [torch.zeros(1, EMBED_DIM)]
            slot_indices = [0]
        x   = torch.cat(tensors, dim=0)
        idx = torch.tensor(slot_indices, dtype=torch.long)
        y   = torch.tensor(self.labels.loc[study, TARGETS].astype(float).values,
                           dtype=torch.float32)
        return study, x, mask, idx, y


def load_state_dict_safe(model, state_dict):
    """
    Load a state_dict into model, transparently handling the '_orig_mod.'
    prefix that torch.compile() adds to every parameter name. Without this,
    loading a compiled model's weights into a fresh (uncompiled) model — or
    vice versa — fails with "Missing/Unexpected key(s)" even though the
    underlying weights are identical.
    """
    sd = state_dict
    model_keys = set(model.state_dict().keys())
    if not any(k in model_keys for k in sd.keys()):
        # keys don't match at all — try stripping/adding the compile prefix
        if all(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        elif not any(k.startswith("_orig_mod.") for k in sd.keys()):
            # model is compiled but checkpoint isn't — add the prefix
            if all(("_orig_mod." + k) in model_keys for k in list(sd.keys())[:1]):
                sd = {"_orig_mod." + k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def auc_mean(y_true, y_pred):
    """
    AUC averaged over targets. y_true may contain continuous weak-label
    scores (from the report parser) instead of strict 0/1 — sklearn's
    roc_auc_score requires binary ground truth, so we threshold at 0.5
    to get a clean binary label for the AUC check. This only affects
    which epoch gets picked as "best" during weak-label pretraining;
    Stage B (fine-tuning on the 58 real studies) always uses exact 0/1
    ground truth, so its reported AUC is unaffected by this.
    """
    y_true_bin = (y_true >= 0.5).astype(np.float32)
    vals = []
    for i in range(len(TARGETS)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            vals.append(roc_auc_score(y_true_bin[:, i], y_pred[:, i]))
    return float(np.mean(vals)) if vals else float("nan")


def train_fold(train_ds, val_ds, device, epochs):
    model   = SlotAttentionModel().to(device)
    # torch.compile gives ~15% speedup on PyTorch 2.0+ (safe, no result change)
    try:
        model = torch.compile(model)
    except Exception:
        pass  # older PyTorch — skip compile
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler  = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    yy      = np.vstack([train_ds.labels.loc[s, TARGETS].astype(float).values
                         for s in train_ds.ids])
    pos     = yy.sum(axis=0)
    pw      = np.maximum((len(yy) - pos) / np.maximum(pos, 1), 1.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    best_state, best_auc = None, -np.inf

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_ds)):
            _, x, mask, idx, y = train_ds.get(i)
            x, mask, idx, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), idx.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                loss = loss_fn(model(x, mask, idx), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                _, x, mask, idx, y = val_ds.get(i)
                P.append(torch.sigmoid(model(x.to(device), mask.to(device),
                                             idx.to(device))).cpu().numpy())
                Y.append(y.numpy())
        auc = auc_mean(np.vstack(Y), np.vstack(P))
        print(f"  epoch {epoch+1:02d}  loss={np.mean(losses):.5f}  val_auc={auc:.5f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model


def finetune_fold(model, train_ds, val_ds, device, epochs, lr=2e-5):
    """
    Same loop as train_fold, but takes an already-initialized model (e.g.
    Stage A weak-label weights) and fine-tunes it with a smaller LR instead
    of training from scratch. Used for Stage B (58 real labels).
    """
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler  = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    yy      = np.vstack([train_ds.labels.loc[s, TARGETS].astype(float).values
                         for s in train_ds.ids])
    pos     = yy.sum(axis=0)
    pw      = np.maximum((len(yy) - pos) / np.maximum(pos, 1), 1.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    best_state, best_auc = None, -np.inf

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_ds)):
            _, x, mask, idx, y = train_ds.get(i)
            x, mask, idx, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), idx.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                loss = loss_fn(model(x, mask, idx), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                _, x, mask, idx, y = val_ds.get(i)
                P.append(torch.sigmoid(model(x.to(device), mask.to(device),
                                             idx.to(device))).cpu().numpy())
                Y.append(y.numpy())
        auc = auc_mean(np.vstack(Y), np.vstack(P))
        print(f"  [finetune] epoch {epoch+1:02d}  loss={np.mean(losses):.5f}  val_auc={auc:.5f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class InferDataset:
    def __init__(self, emb_df):
        self.emb_df = emb_df
        self.ids    = sorted(emb_df["StudyInstanceUID"].unique())

    def __len__(self): return len(self.ids)

    def get(self, i):
        study = self.ids[i]
        rows  = self.emb_df[self.emb_df["StudyInstanceUID"] == study]
        slot_to_file = {r["slot_name"]: r["embedding_file"]
                        for _, r in rows.iterrows() if r["presence_mask"] == 1}
        tensors, slot_indices, mask = [], [], torch.zeros(N_SLOT)
        for s_idx, slot_name in enumerate(SLOT_NAMES):
            if slot_name in slot_to_file:
                try:
                    x = load_embedding(slot_to_file[slot_name])
                    tensors.append(x)
                    slot_indices.extend([s_idx] * len(x))
                    mask[s_idx] = 1.0
                except Exception:
                    pass
        if not tensors:
            tensors = [torch.zeros(1, EMBED_DIM)]
            slot_indices = [0]
        return study, torch.cat(tensors, dim=0), mask, torch.tensor(slot_indices, dtype=torch.long)


def run_inference(emb_df, model_paths, device):
    ds      = InferDataset(emb_df)
    all_preds = np.zeros((len(ds), len(TARGETS)), dtype=np.float32)

    for mp in model_paths:
        model = SlotAttentionModel().to(device)
        ckpt  = torch.load(mp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds = np.zeros((len(ds), len(TARGETS)), dtype=np.float32)
        with torch.no_grad():
            for i in range(len(ds)):
                study, x, mask, idx = ds.get(i)
                logits = model(x.to(device), mask.to(device), idx.to(device))
                preds[i] = torch.sigmoid(logits).cpu().numpy()
        all_preds += preds / len(model_paths)
        print(f"  Applied: {Path(mp).name}")

    study_ids = ds.ids
    return study_ids, all_preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST STACKING LAYER
#
# Second, structurally-different model trained on the SAME cached MedSigLIP
# embeddings the neural attention model already uses (no re-encoding needed).
# Trees make different mistakes than the attention model, so blending the
# two typically adds a small but real AUC gain from diversity alone.
#
# Pipeline:
#   pooled embedding per study (mean over all present slots/slices)
#     -> PCA to PCA_COMPONENTS dims (fit once on the Stage A pool)
#     -> + slot presence mask
#     -> Stage A: 12 per-disease XGBoost models, 5-fold CV on weak+real pool
#     -> Stage A out-of-fold predictions become extra meta-features
#     -> Stage B: 12 per-disease XGBoost models, 5-fold CV on the 58 real
#        studies only (same folds as the neural net's Stage B loop)
#     -> blend weight (alpha) picked automatically from OOF AUC
#
# Every block below is wrapped so that any failure (missing package,
# degenerate fold, etc.) disables the stacking layer and falls back to
# NN-only predictions — this addition can only help or do nothing, never
# make the submission worse than before.
# ══════════════════════════════════════════════════════════════════════════════
XGB_AVAILABLE = True
try:
    import xgboost as xgb
except ImportError:
    try:
        print("Installing xgboost...")
        subprocess.run(["pip", "install", "xgboost", "-q"], check=True)
        import xgboost as xgb
    except Exception as e:
        print(f"[WARN] xgboost unavailable ({e}) — stacking layer disabled.")
        XGB_AVAILABLE = False

PCA_COMPONENTS = 64
XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=3,
    reg_lambda=2.0, reg_alpha=0.5, eval_metric="auc",
    tree_method="hist", verbosity=0,
)
XGB_N_ESTIMATORS = 300
ALPHA_CANDIDATES = [1.0, 0.85, 0.70, 0.55, 0.40]   # 1.0 = NN only, fallback-safe


def study_pooled_embedding(study, emb_df):
    """
    Mean-pool ALL slice embeddings for one study across all present slots.
    Reuses the exact same cached .pt files the neural net loads — zero
    extra MedSigLIP forward passes needed.
    Returns (EMBED_DIM,) vector + (N_SLOT,) presence mask.
    """
    rows = emb_df[emb_df["StudyInstanceUID"] == study]
    slot_to_file = {r["slot_name"]: r["embedding_file"]
                    for _, r in rows.iterrows() if r["presence_mask"] == 1}
    tensors = []
    mask = np.zeros(N_SLOT, dtype=np.float32)
    for s_idx, slot_name in enumerate(SLOT_NAMES):
        if slot_name in slot_to_file:
            try:
                tensors.append(load_embedding(slot_to_file[slot_name]))
                mask[s_idx] = 1.0
            except Exception:
                pass
    if not tensors:
        return np.zeros(EMBED_DIM, dtype=np.float32), mask
    return torch.cat(tensors, dim=0).mean(dim=0).numpy(), mask


def build_tabular_matrix(study_ids, emb_df):
    """[n, EMBED_DIM] pooled embeddings + [n, N_SLOT] presence masks, in order."""
    feats, masks = [], []
    for s in study_ids:
        f, m = study_pooled_embedding(s, emb_df)
        feats.append(f)
        masks.append(m)
    return np.stack(feats).astype(np.float32), np.stack(masks).astype(np.float32)


def fit_pca(X, n_components=PCA_COMPONENTS):
    n_components = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return pca


def make_features(raw_embed, mask, pca, extra=None):
    """PCA-reduce pooled embeddings, concat presence mask + optional extra meta-features."""
    reduced = pca.transform(raw_embed)
    parts = [reduced, mask]
    if extra is not None:
        parts.append(extra)
    return np.concatenate(parts, axis=1).astype(np.float32)


def soft_label_weight(y_soft, is_real):
    """
    Confidence weight per (study, target) pair.
      Real (official) label -> weight 3.0, full trust.
      Weak (parsed) label   -> weight 0.25 to 1.0, scaled by how far the
                                soft label sits from the undecided 0.5 point
                                (values near 0.5 mean the parser was unsure).
    """
    w = np.where(
        is_real[:, None],
        3.0,
        0.25 + 0.75 * (2.0 * np.abs(y_soft - 0.5)).clip(0, 1),
    )
    return w.astype(np.float32)


def train_xgb_per_target(X, Y, W):
    """
    One XGBoost binary classifier per disease target.
    Y is thresholded at 0.5 for the binary label; W is the per-sample,
    per-target confidence weight from soft_label_weight().
    Targets with fewer than 2 classes in this fold are skipped (returns
    None for that target — predict_xgb() falls back to 0.5 for it).
    """
    models = {}
    for t_idx, target in enumerate(TARGETS):
        y = (Y[:, t_idx] >= 0.5).astype(int)
        w = W[:, t_idx]
        if len(np.unique(y)) < 2:
            models[target] = None
            continue
        dtrain = xgb.DMatrix(X, label=y, weight=w)
        models[target] = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=XGB_N_ESTIMATORS,
            verbose_eval=False,
        )
    return models


def predict_xgb(models, X):
    """[n, N_TARGETS] predicted probabilities; 0.5 for any skipped target."""
    n = X.shape[0]
    P = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    dtest = xgb.DMatrix(X)
    for t_idx, target in enumerate(TARGETS):
        m = models.get(target)
        if m is not None:
            P[:, t_idx] = m.predict(dtest)
    return P


def main():
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # local flag (not the module-level XGB_AVAILABLE) so any failure inside
    # main() can disable the stacking layer for this run only, cleanly
    xgb_ok = XGB_AVAILABLE

    print("=" * 60)
    print("RSNA KNEE ABNORMALITY DETECTION")
    print(f"Device  : {device}")
    print(f"Kaggle  : {IS_KAGGLE}")
    print("=" * 60)

    # ── load labels: parser output, NOT train.csv ──────────────────
    test_df         = pd.read_csv(DATA_ROOT / "test.csv")
    test_series_csv  = DATA_ROOT / "test_series.csv"
    train_series_csv = DATA_ROOT / "train_series.csv"

    print(f"Using parsed labels: {PARSED_LABELS_CSV}")
    parsed = pd.read_csv(PARSED_LABELS_CSV, dtype={"StudyInstanceUID": str})
    parsed[TARGETS] = parsed[TARGETS].fillna(0)
    if "is_real_label" not in parsed.columns:
        raise RuntimeError(
            f"{PARSED_LABELS_CSV} has no 'is_real_label' column — "
            "run train_and_predict.py first to produce it."
        )

    real_df = parsed[parsed["is_real_label"]].copy()
    weak_df = parsed[~parsed["is_real_label"]].copy()
    print(f"Real-labeled studies (true 0/1): {len(real_df)}")
    print(f"Weak-labeled studies (parser)  : {len(weak_df)}")

    n_weak_needed = max(0, N_TOTAL_STUDIES - len(real_df))
    if n_weak_needed > len(weak_df):
        print(f"[WARN] Only {len(weak_df)} weak-labeled studies available, "
              f"wanted {n_weak_needed} — using all of them.")
        n_weak_needed = len(weak_df)
    weak_sample = weak_df.sample(n=n_weak_needed, random_state=SAMPLE_SEED)

    labeled = pd.concat([real_df, weak_sample], ignore_index=True)
    print(f"Training pool total            : {len(labeled)} "
          f"({len(real_df)} real + {len(weak_sample)} weak)")
    print(f"Test studies    : {len(test_df)}")

    # ══ TRAIN PIPELINE ══════════════════════════════════════════
    if IS_KAGGLE:
        print("\n── TRAIN: On Kaggle — data already unzipped from attached Dataset, skipping unzip ──")
    else:
        print("\n── TRAIN: Ensure studies unzipped ──")
        ensure_studies_unzipped(labeled["StudyInstanceUID"], TRAIN_SERIES, TRAIN_SERIES_ZIP)

    print("\n── TRAIN: Scan DICOMs ──")
    _cached_dcm_idx = WORK_DIR / "train_dicom_index.csv"
    if _cached_dcm_idx.exists():
        print(f"  Reusing cached DICOM index: {_cached_dcm_idx}")
        train_dcm = pd.read_csv(_cached_dcm_idx, dtype=str)
    else:
        train_study_filter = set(labeled["StudyInstanceUID"].astype(str).tolist())
        train_dcm = scan_dicoms(TRAIN_SERIES, train_series_csv, study_filter=train_study_filter)
        train_dcm.to_csv(_cached_dcm_idx, index=False)

    print("\n── TRAIN: Build slots ──")
    _cached_slots = WORK_DIR / "train_slots_cache.pkl"
    if _cached_slots.exists():
        print(f"  Reusing cached slots: {_cached_slots}")
        with open(_cached_slots, "rb") as f:
            train_slots, train_lat = pickle.load(f)
    else:
        train_lat   = detect_laterality(train_dcm)
        train_slots = assign_slots(train_dcm)
        train_slots["laterality"] = train_slots["StudyInstanceUID"].map(train_lat)
        with open(_cached_slots, "wb") as f:
            pickle.dump((train_slots, train_lat), f)
        print(f"  Cached slots to: {_cached_slots}")

    print("\n── TRAIN: Embed ──")
    processor, model_enc, device_enc = load_medsiglip()
    train_emb_idx = embed_slots(train_slots, train_dcm, processor, model_enc,
                                 device_enc, train_lat, EMB_DIR / "train")
    train_emb_idx.to_csv(WORK_DIR / "train_embedding_index.csv", index=False)
    del model_enc; torch.cuda.empty_cache()

    # ══ STAGE A — PRETRAIN ON WEAK LABELS (all 4340: 58 real + weak) ═══
    print("\n── STAGE A: Pretrain on weak labels (4340 studies) ──")
    labeled_ids = set(labeled["StudyInstanceUID"].astype(str))
    emb_ids     = set(train_emb_idx["StudyInstanceUID"].astype(str))
    common      = labeled_ids & emb_ids
    print(f"Studies with labels + embeddings: {len(common)}")

    if len(common) < 2:
        print("[STOP] Not enough labeled studies.")
        return

    lbl = labeled[labeled["StudyInstanceUID"].astype(str).isin(common)].copy()
    emb = train_emb_idx[train_emb_idx["StudyInstanceUID"].astype(str).isin(common)].copy()

    real_ids_common = set(real_df["StudyInstanceUID"].astype(str)) & common
    pretrain_ids = np.array(sorted(common))
    print(f"Pretrain pool: {len(pretrain_ids)} studies "
          f"({len(real_ids_common)} of them real-labeled)")

    # Stage A: 5-fold CV on all weak+real studies
    # Best fold model (highest val AUC) used as starting point for Stage B
    #
    # RESUME SUPPORT: if stage_a_pretrained.pt already exists (either from
    # this session or copied in from a re-attached Kaggle Dataset of a
    # previous run's /kaggle/working/models folder), skip Stage A entirely
    # and go straight to Stage B using those weights.
    pretrain_path = MODEL_DIR / "stage_a_pretrained.pt"
    if pretrain_path.exists():
        print(f"\n── STAGE A: Found existing checkpoint at {pretrain_path} — skipping Stage A training ──")
        ckpt = torch.load(pretrain_path, map_location=device)
        pretrained_model = SlotAttentionModel().to(device)
        pretrained_model = load_state_dict_safe(pretrained_model, ckpt["model_state_dict"])
        best_stage_a_auc = None
    else:
        print(f"Stage A: 5-fold CV on {len(pretrain_ids)} studies")
        stage_a_table  = pd.DataFrame({"study": pretrain_ids})
        stage_a_gkf    = GroupKFold(n_splits=5)
        stage_a_models = []
        best_stage_a_auc   = -1
        best_stage_a_model = None

        for sa_fold, (sa_tri, sa_vi) in enumerate(
            stage_a_gkf.split(stage_a_table, groups=stage_a_table.study), 1
        ):
            sa_tr_ids = pretrain_ids[sa_tri]
            sa_va_ids = pretrain_ids[sa_vi]
            pre_tr_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_tr_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_tr_ids)])
            pre_va_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_va_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_va_ids)])
            print(f"\nStage A FOLD {sa_fold}  train={len(pre_tr_ds)}  val={len(pre_va_ds)}")

            sa_model = train_fold(pre_tr_ds, pre_va_ds, device, epochs=30)

            # evaluate this fold
            sa_model.eval()
            sa_Y, sa_P = [], []
            with torch.no_grad():
                for i in range(len(pre_va_ds)):
                    _, x, mask, idx, y = pre_va_ds.get(i)
                    pred = torch.sigmoid(sa_model(x.to(device), mask.to(device),
                                                  idx.to(device))).cpu().numpy()
                    sa_Y.append(y.numpy()); sa_P.append(pred)
            sa_auc = auc_mean(np.vstack(sa_Y), np.vstack(sa_P))
            print(f"[Stage A FOLD {sa_fold}] val_auc={sa_auc:.5f}")

            # save each fold model
            sa_path = MODEL_DIR / f"stage_a_fold_{sa_fold}.pt"
            torch.save({"model_state_dict": sa_model.state_dict(),
                        "targets": TARGETS, "slot_names": SLOT_NAMES,
                        "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM}, sa_path)
            stage_a_models.append(sa_model)

            if not np.isnan(sa_auc) and sa_auc > best_stage_a_auc:
                best_stage_a_auc   = sa_auc
                best_stage_a_model = sa_model

        # use best Stage A fold as pretrained_model for Stage B
        pretrained_model = best_stage_a_model
        torch.save({"model_state_dict": pretrained_model.state_dict(),
                    "targets": TARGETS, "slot_names": SLOT_NAMES,
                    "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM},
                   pretrain_path)
        print(f"\nBest Stage A AUC: {best_stage_a_auc:.5f}")
        print(f"Saved best Stage A weights: {pretrain_path}")

    # ══ STAGE A (XGBOOST) — SAME WEAK+REAL POOL, TREE-BASED MODEL ═══════
    xgb_pca              = None
    xgb_stage_a_models   = []      # one dict-of-boosters per fold, for test-time averaging
    stage_a_xgb_oof      = None    # [len(pretrain_ids), N_TARGETS], meta-feature for Stage B
    if xgb_ok:
        try:
            xgb_stage_a_path = MODEL_DIR / "xgb_stage_a.pkl"
            print("\n── STAGE A (XGBoost): pooling embeddings + PCA ──")
            raw_embed, presence = build_tabular_matrix(pretrain_ids, emb)
            xgb_pca = fit_pca(raw_embed)
            print(f"  PCA: {raw_embed.shape[1]} -> {xgb_pca.n_components_} dims "
                  f"(explained var {xgb_pca.explained_variance_ratio_.sum():.2%})")

            Y_pool = lbl.set_index("StudyInstanceUID").loc[pretrain_ids, TARGETS].values.astype(np.float32)
            is_real_pool = np.array([sid in real_ids_common for sid in pretrain_ids])
            W_pool = soft_label_weight(Y_pool, is_real_pool)
            X_pool = make_features(raw_embed, presence, xgb_pca)

            if xgb_stage_a_path.exists():
                print(f"  Found existing checkpoint at {xgb_stage_a_path} — skipping Stage A XGBoost training")
                with open(xgb_stage_a_path, "rb") as f:
                    saved = pickle.load(f)
                xgb_pca            = saved["pca"]
                xgb_stage_a_models = saved["fold_models"]
                stage_a_xgb_oof    = saved["oof"]
                X_pool = make_features(raw_embed, presence, xgb_pca)  # rebuild with loaded pca (should match)
            else:
                stage_a_xgb_oof = np.zeros((len(pretrain_ids), len(TARGETS)), dtype=np.float32)
                xgb_sa_gkf = GroupKFold(n_splits=5)
                xgb_sa_table = pd.DataFrame({"study": pretrain_ids})
                for xfold, (xtri, xvi) in enumerate(
                    xgb_sa_gkf.split(xgb_sa_table, groups=xgb_sa_table.study), 1
                ):
                    fold_models = train_xgb_per_target(X_pool[xtri], Y_pool[xtri], W_pool[xtri])
                    stage_a_xgb_oof[xvi] = predict_xgb(fold_models, X_pool[xvi])
                    xgb_stage_a_models.append(fold_models)
                    print(f"  [Stage A XGBoost FOLD {xfold}] trained "
                          f"({sum(m is not None for m in fold_models.values())}/{len(TARGETS)} targets)")

                sa_xgb_auc = auc_mean(Y_pool, stage_a_xgb_oof)
                print(f"  Stage A XGBoost OOF mean AUC: {sa_xgb_auc:.5f}")

                with open(xgb_stage_a_path, "wb") as f:
                    pickle.dump({"pca": xgb_pca, "fold_models": xgb_stage_a_models,
                                "oof": stage_a_xgb_oof}, f)
                print(f"  Saved: {xgb_stage_a_path}")
        except Exception as e:
            print(f"[WARN] Stage A XGBoost failed ({e}) — stacking layer disabled for this run.")
            xgb_ok = False

    # ══ STAGE B — FINE-TUNE ON THE 58 REAL LABELS (5-fold CV) ══════════
    print("\n── STAGE B: Fine-tune on 58 real-labeled studies ──")
    real_lbl = lbl[lbl["StudyInstanceUID"].isin(real_ids_common)].copy()
    real_emb = emb[emb["StudyInstanceUID"].isin(real_ids_common)].copy()

    ids      = np.array(sorted(real_ids_common))
    table    = pd.DataFrame({"study": ids})
    n_splits = min(5, len(ids))
    gkf      = GroupKFold(n_splits=n_splits)
    oof      = np.zeros((len(ids), len(TARGETS)), dtype=np.float32)

    # ── XGBoost Stage B: build features once, train per-fold inside the loop below ──
    xgb_stage_b_models = []
    xgb_oof            = None
    if xgb_ok and xgb_pca is not None:
        try:
            raw_embed_b, presence_b = build_tabular_matrix(ids, real_emb)
            # Stage A XGBoost OOF predictions, reordered to match `ids`, used as
            # meta-features — this is the actual "stacking" part: Stage B trees
            # get to see what the Stage A model (trained 4340 studies) thought.
            pretrain_pos = {sid: i for i, sid in enumerate(pretrain_ids)}
            meta_b = np.stack([stage_a_xgb_oof[pretrain_pos[sid]] for sid in ids]).astype(np.float32)
            X_stage_b = make_features(raw_embed_b, presence_b, xgb_pca, extra=meta_b)
            Y_stage_b = real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values.astype(np.float32)
            xgb_oof   = np.zeros((len(ids), len(TARGETS)), dtype=np.float32)
        except Exception as e:
            print(f"[WARN] XGBoost Stage B feature build failed ({e}) — stacking layer disabled for this run.")
            xgb_ok = False

    for fold, (tri, vi) in enumerate(gkf.split(table, groups=table.study), 1):
        fold_ckpt_path = MODEL_DIR / f"fold_{fold}.pt"
        tr_fold_ids = ids[tri]; va_fold_ids = ids[vi]
        tr_ds = StudyDataset(real_emb[real_emb["StudyInstanceUID"].isin(tr_fold_ids)],
                              real_lbl[real_lbl["StudyInstanceUID"].isin(tr_fold_ids)])
        va_ds = StudyDataset(real_emb[real_emb["StudyInstanceUID"].isin(va_fold_ids)],
                              real_lbl[real_lbl["StudyInstanceUID"].isin(va_fold_ids)])

        if fold_ckpt_path.exists():
            print(f"\nFOLD {fold}  — checkpoint already exists at {fold_ckpt_path}, "
                  f"loading it and skipping training for this fold")
            ckpt = torch.load(fold_ckpt_path, map_location=device)
            fold_model = SlotAttentionModel().to(device)
            fold_model = load_state_dict_safe(fold_model, ckpt["model_state_dict"])
        else:
            print(f"\nFOLD {fold}  train={len(tr_ds)}  val={len(va_ds)}")
            # start from Stage A (weak-label pretrained) weights, then fine-tune
            fold_model = SlotAttentionModel().to(device)
            fold_model = load_state_dict_safe(fold_model, pretrained_model.state_dict())
            fold_model = finetune_fold(fold_model, tr_ds, va_ds, device, epochs=15, lr=2e-5)

        fold_model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(va_ds)):
                study, x, mask, idx, y = va_ds.get(i)
                pred = torch.sigmoid(fold_model(x.to(device), mask.to(device),
                                                idx.to(device))).cpu().numpy()
                oof[np.where(ids == study)[0][0]] = pred
                Y.append(y.numpy()); P.append(pred)

        fold_auc = auc_mean(np.vstack(Y), np.vstack(P))
        print(f"[FOLD {fold}] AUC={fold_auc:.5f}")
        torch.save({"model_state_dict": fold_model.state_dict(),
                    "targets": TARGETS, "slot_names": SLOT_NAMES,
                    "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM},
                   MODEL_DIR / f"fold_{fold}.pt")

        # ── XGBoost Stage B: same fold split, tree model on pooled+PCA features ──
        if xgb_ok and xgb_oof is not None:
            try:
                xgb_fold_path = MODEL_DIR / f"xgb_fold_{fold}.pkl"
                if xgb_fold_path.exists():
                    with open(xgb_fold_path, "rb") as f:
                        fold_xgb_models = pickle.load(f)
                    print(f"  [XGB FOLD {fold}] loaded existing checkpoint")
                else:
                    fold_xgb_models = train_xgb_per_target(
                        X_stage_b[tri], Y_stage_b[tri],
                        np.ones_like(Y_stage_b[tri])   # all real labels, equal trust
                    )
                    with open(xgb_fold_path, "wb") as f:
                        pickle.dump(fold_xgb_models, f)
                xgb_oof[vi] = predict_xgb(fold_xgb_models, X_stage_b[vi])
                xgb_stage_b_models.append(fold_xgb_models)
                xgb_fold_auc = auc_mean(Y_stage_b[vi], xgb_oof[vi])
                print(f"  [XGB FOLD {fold}] AUC={xgb_fold_auc:.5f}")
            except Exception as e:
                print(f"[WARN] XGBoost Stage B fold {fold} failed ({e}) — "
                      f"stacking layer disabled for remaining folds.")
                xgb_ok = False

    oof_df = pd.DataFrame(oof, columns=TARGETS)
    oof_df.insert(0, "StudyInstanceUID", ids)
    oof_df.to_csv(MODEL_DIR / "oof_predictions.csv", index=False)
    mean_auc = auc_mean(real_lbl[TARGETS].values, oof)
    print(f"\nOOF Mean AUC (fine-tuned, on 58 real labels): {mean_auc:.5f}")
    print("\nPer-target AUC:")
    for i, t in enumerate(TARGETS):
        col = real_lbl[TARGETS].values[:, i]
        if len(np.unique(col)) > 1:
            a = roc_auc_score(col, oof[:, i])
            print(f"  {t:22s}: {a:.4f}")
        else:
            print(f"  {t:22s}: (no positives in OOF)")

    # ── Pick blend weight (alpha) between NN and XGBoost using OOF AUC ──
    # alpha=1.0 means NN-only — always in the candidate list, so blending
    # can never be chosen over the plain NN unless it genuinely scores higher.
    best_alpha = 1.0
    if xgb_ok and xgb_oof is not None:
        try:
            xgb_only_auc = auc_mean(Y_stage_b, xgb_oof)
            print(f"\nXGBoost-only OOF AUC: {xgb_only_auc:.5f}")
            print("\nBlend search (alpha = NN weight, 1-alpha = XGBoost weight):")
            best_blend_auc = mean_auc
            for a in ALPHA_CANDIDATES:
                blended = a * oof + (1 - a) * xgb_oof
                blend_auc = auc_mean(Y_stage_b, blended)
                marker = ""
                if not np.isnan(blend_auc) and blend_auc > best_blend_auc:
                    best_blend_auc = blend_auc
                    best_alpha     = a
                    marker = "  <- best so far"
                print(f"  alpha={a:.2f}  blended_auc={blend_auc:.5f}{marker}")
            print(f"\nChosen alpha={best_alpha:.2f}  "
                  f"(NN-only={mean_auc:.5f} -> blended={best_blend_auc:.5f})")
        except Exception as e:
            print(f"[WARN] Blend search failed ({e}) — using NN-only predictions.")
            best_alpha = 1.0

    # ══ TEST PIPELINE ═══════════════════════════════════════════
    print("\n── TEST: Scan DICOMs ──")
    test_dcm = scan_dicoms(TEST_SERIES, test_series_csv)

    print("\n── TEST: Build slots ──")
    test_lat   = detect_laterality(test_dcm)
    test_slots = assign_slots(test_dcm)
    test_slots["laterality"] = test_slots["StudyInstanceUID"].map(test_lat)

    print("\n── TEST: Embed ──")
    processor, model_enc, device_enc = load_medsiglip()
    test_emb_idx = embed_slots(test_slots, test_dcm, processor, model_enc,
                                device_enc, test_lat, EMB_DIR / "test")
    del model_enc; torch.cuda.empty_cache()

    print("\n── TEST: Inference ──")
    model_paths = sorted(MODEL_DIR.glob("fold_*.pt"))
    study_ids, preds = run_inference(test_emb_idx, model_paths, device)

    # ── TEST: XGBoost stacking prediction + blend with NN ──
    final_preds = preds
    if xgb_ok and best_alpha < 1.0 and xgb_pca is not None and xgb_stage_b_models:
        try:
            print("\n── TEST: XGBoost stacking prediction ──")
            raw_embed_t, presence_t = build_tabular_matrix(study_ids, test_emb_idx)

            # Stage A meta-features for test studies — average across Stage A folds
            X_stage_a_t = make_features(raw_embed_t, presence_t, xgb_pca)
            meta_t = np.zeros((len(study_ids), len(TARGETS)), dtype=np.float32)
            for fold_models in xgb_stage_a_models:
                meta_t += predict_xgb(fold_models, X_stage_a_t) / len(xgb_stage_a_models)

            # Stage B prediction — average across Stage B folds
            X_stage_b_t = make_features(raw_embed_t, presence_t, xgb_pca, extra=meta_t)
            xgb_test_preds = np.zeros((len(study_ids), len(TARGETS)), dtype=np.float32)
            for fold_models in xgb_stage_b_models:
                xgb_test_preds += predict_xgb(fold_models, X_stage_b_t) / len(xgb_stage_b_models)

            final_preds = best_alpha * preds + (1 - best_alpha) * xgb_test_preds
            print(f"  Blended NN + XGBoost predictions (alpha={best_alpha:.2f})")
        except Exception as e:
            print(f"[WARN] XGBoost test-time prediction failed ({e}) — using NN-only predictions.")
            final_preds = preds

    # ══ SUBMISSION ══════════════════════════════════════════════
    sub = pd.DataFrame(final_preds, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", study_ids)

    # add any test studies with no embeddings at 0.5 (default)
    missing = set(test_df["StudyInstanceUID"].astype(str)) - set(study_ids)
    if missing:
        print(f"[WARN] {len(missing)} test studies had no embeddings — defaulting to 0.5")
        filler = pd.DataFrame([[sid] + [0.5]*len(TARGETS) for sid in missing],
                               columns=["StudyInstanceUID"] + TARGETS)
        sub = pd.concat([sub, filler], ignore_index=True)

    # reorder to match sample submission
    sub = sub.set_index("StudyInstanceUID").reindex(
        test_df["StudyInstanceUID"].astype(str)).reset_index()
    sub.columns = ["StudyInstanceUID"] + TARGETS

    out = WORK_DIR / "submission.csv"
    sub.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"OOF AUC    : {mean_auc:.5f}")
    print(f"Submission : {out}")
    print(f"Shape      : {sub.shape}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
