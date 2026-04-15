import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import os
import re
import json
import time
import random
import base64
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np  # ✅ for CLAHE-like local contrast

# ======================================================
# 0) API Configuration (fixed model: GPT-4o)
# ======================================================
# ✅ IMPORTANT: Do NOT hardcode real API keys in source code.
# Set your key via environment variable instead:
#   export LUXIA_API_KEY="YOUR_KEY"
# or on Windows PowerShell:
#   setx LUXIA_API_KEY "YOUR_KEY"
API_KEY = os.environ.get("OPENAI_API_KEY", "") 
BRIDGE_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5-nano"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# ======================================================
# 1) I/O Paths
# ======================================================
TEST_CSV_PATH = "./test.csv"            # columns: id,img_url
OUT_SUBMISSION = "./submission.csv"    # columns: id,label
OUT_EXPLAIN = "./explanations.csv"     # columns: id,label,confidence,used_vote,valid,reason,checks,vote_dist,roi_meta

LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# ======================================================
# ✅ Input image standardization / enhancement config
# ======================================================
RESIZE_TO = (224, 224)

# Preprocessing options:
# - First run: resize only
# - Fallback run: CLAHE-like local contrast + sharpen
USE_ENHANCE_DEFAULT = False
USE_ENHANCE_FALLBACK = True

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
SHARPEN_FACTOR = 1.8

# ======================================================
# 2) ROI / Sub-ROI Settings
# ======================================================
ROI_X1 = 0.18
ROI_X2 = 0.82
ROI_Y1 = 0.45
ROI_Y2 = 0.98

HOLE_CENTERS = [0.18, 0.50, 0.82]
HOLE_Y_CENTER = 0.78
HOLE_BOX_W = 0.30
HOLE_BOX_H = 0.45

# ======================================================
# 3) System / Prompts (English)
# ======================================================
SYSTEM = (
    "You are an inspector for semiconductor device visual inspection in a manufacturing process.\n"
    "You MUST output only a valid JSON object. Do NOT output any other sentences, explanations, or code blocks.\n"
)

PROMPT = r"""
Input images (5):
1) FULL: the original full image (package damage MUST be judged using ONLY this image)
2) ANNOTATED: the original image with ROI (red) + three Sub-ROIs (green) drawn (for location reference)
3) Sub-ROI #1: crop around the LEFT hole (for lead connection judgment)
4) Sub-ROI #2: crop around the CENTER hole (for lead connection judgment)
5) Sub-ROI #3: crop around the RIGHT hole (for lead connection judgment)

[What you must judge]
A) package_intact:
- From FULL (the full original image), return true if there is NO chipping, cracking, breakage, or damaged corners/edges
  on the package (black body).
- Return false if any damage is visible.
- For package judgment, DO NOT use Sub-ROIs; use FULL only.

B) hole1_status, hole2_status, hole3_status:
- From Sub-ROI #1~#3, decide whether each hole is "connected" to the lead.

[Definition of lead connection (connected) — VERY IMPORTANT]
- connected:
  The metal lead directly touches the hole (black circle) at the TOP (12 o'clock position),
  and the lead descends into the hole (top-to-bottom) with NO visible vertical gap between lead and hole.
- not_connected:
  The lead is not positioned at the top of the hole, OR
  there is a clear empty space (gap) between the lead and the hole.
  (Side touch / diagonal brushing / being close is NOT considered connected.)
- uncertain:
  If resolution/noise prevents determining 12 o'clock contact or the presence/absence of a gap.

[Reason writing rules — MUST follow]
The reason MUST include BOTH:
(1) Package: whether it is damaged + evidence based on FULL (e.g., top/corners/edges intact)
(2) Leads: for hole1~3, evidence for BOTH "12 o'clock contact" AND "gap presence/absence"

Example:
"Package: no damage (based on FULL). hole1: 12 o'clock contact + no gap = connected / hole2: gap present = not_connected / hole3: ..."

[Output JSON format]
{
  "package_intact": true,
  "hole1_status": "connected",
  "hole2_status": "connected",
  "hole3_status": "connected",
  "confidence": 0.0,
  "reason": ""
}
""".strip()

# ======================================================
# ✅ CoVe (run only when any hole is "uncertain")
# ======================================================
COVE_Q_SYSTEM = (
    "You are a 'verification-question generator' for manufacturing visual inspection decisions.\n"
    "You MUST output valid JSON only. Do NOT output any other sentences or explanations.\n"
)

COVE_Q_PROMPT = r"""
Your task:
- Create a short checklist of verification questions to validate the draft decision below.
- For each hole status, questions MUST explicitly check BOTH:
  (1) whether there is contact at the 12 o'clock position, and
  (2) whether a gap exists.
- For package_intact, questions MUST enforce checking ONLY the FULL image.
- Keep questions short and clear.

[Output JSON format]
{
  "questions": [
    {"key": "package_intact", "q": "..."},
    {"key": "hole1_status", "q": "..."},
    {"key": "hole2_status", "q": "..."},
    {"key": "hole3_status", "q": "..."}
  ]
}

[draft]
{draft_json}
""".strip()

COVE_VERIFY_SYSTEM = (
    "You are an inspector for semiconductor device visual inspection in a manufacturing process, and you are now in the 'verification' stage.\n"
    "You MUST output only a valid JSON object. Do NOT output any other sentences, explanations, or code blocks.\n"
    "Do NOT be anchored to the draft decision; re-judge independently based only on image evidence.\n"
)

COVE_VERIFY_PROMPT = r"""
Input images (5):
1) FULL: the original full image (package damage MUST be judged using ONLY this image)
2) ANNOTATED: ROI (red) + Sub-ROIs (green) overlay for location reference
3) Sub-ROI #1
4) Sub-ROI #2
5) Sub-ROI #3

Check each checklist question below, then output ONLY the final decision JSON.

[Checklist]
{questions_json}

[Judgment rules]
- package_intact: using FULL only, true if NO chipping/cracking/breakage/damaged corners, false otherwise
- hole*_status: connected / not_connected / uncertain (use the same definitions as before)
- reason MUST include:
  (1) Package: evidence based on FULL
  (2) hole1~3: evidence for BOTH 12 o'clock contact AND gap presence/absence

[Output JSON format]
{
  "package_intact": true,
  "hole1_status": "connected",
  "hole2_status": "connected",
  "hole3_status": "connected",
  "confidence": 0.0,
  "reason": ""
}
""".strip()

# ======================================================
# 4) Utilities
# ======================================================
def _post_chat(messages, timeout=90, max_retries=3):
    """POST to the chat-completions bridge with basic retry + backoff."""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL,
                "messages": messages,
                "max_completion_tokens": 4096,
            }
            r = requests.post(BRIDGE_URL, headers=HEADERS, json=payload, timeout=timeout)
            if r.status_code != 200:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise RuntimeError(f"status={r.status_code}, body={r.text[:300]}")
            return r.json()["choices"][0]["message"]["content"].strip()
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(0.5 * (2 ** attempt) + random.uniform(0, 0.2))
                continue
            raise


def _is_refusal_text(s: str) -> bool:
    s_low = s.strip().lower()
    return ("i'm sorry" in s_low and "can't assist" in s_low) or ("i can't assist" in s_low)


def _safe_json_extract(s: str) -> dict:
    """Parse JSON robustly even if the model accidentally wraps extra text."""
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.S)
        if m:
            return json.loads(m.group(0))
    raise ValueError(f"JSON parse failed: {s[:200]}")


def _download_image_bytes(img_url: str, timeout=30) -> bytes:
    r = requests.get(img_url, timeout=timeout)
    r.raise_for_status()
    return r.content


def _pil_to_b64_png(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(1, min(W, x2))
    y2 = max(1, min(H, y2))
    if x2 <= x1 + 1:
        x2 = min(W, x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(H, y1 + 2)
    return x1, y1, x2, y2


def _norm_status(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("connected", "not_connected", "uncertain"):
        return s
    if s in ("contact", "touch", "touching", "connected_to", "ok"):
        return "connected"
    if s in ("disconnected", "no_contact", "not contact", "notconnected", "not-connected", "ng"):
        return "not_connected"
    return "uncertain"


# ======================================================
# ✅ Resize + (optional) local contrast + sharpen
# ======================================================
def _resize_224(img: Image.Image) -> Image.Image:
    return img.resize(RESIZE_TO, resample=Image.BICUBIC)


def _clahe_gray_np(gray_u8: np.ndarray, clip_limit: float = 2.0, grid=(8, 8)) -> np.ndarray:
    """
    A simple CLAHE-like implementation in NumPy.
    (Not OpenCV CLAHE, but similar intent: local histogram equalization with clipping.)
    """
    H, W = gray_u8.shape
    gh, gw = grid
    tile_h = max(1, H // gh)
    tile_w = max(1, W // gw)

    out = np.zeros_like(gray_u8, dtype=np.float32)

    luts = [[None for _ in range(gw)] for _ in range(gh)]
    for ty in range(gh):
        for tx in range(gw):
            y1 = ty * tile_h
            x1 = tx * tile_w
            y2 = H if ty == gh - 1 else (ty + 1) * tile_h
            x2 = W if tx == gw - 1 else (tx + 1) * tile_w
            tile = gray_u8[y1:y2, x1:x2]

            hist, _ = np.histogram(tile.flatten(), bins=256, range=(0, 256))
            clip_val = int(clip_limit * (tile.size / 256.0))
            excess = np.maximum(hist - clip_val, 0).sum()
            hist = np.minimum(hist, clip_val)
            hist += excess // 256

            cdf = hist.cumsum()
            cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-12)
            lut = np.floor(255 * cdf).astype(np.uint8)
            luts[ty][tx] = lut

    for y in range(H):
        ty = min(gh - 1, y // tile_h)
        ty2 = min(gh - 1, ty + 1)
        wy = (y - ty * tile_h) / (tile_h + 1e-12)
        for x in range(W):
            tx = min(gw - 1, x // tile_w)
            tx2 = min(gw - 1, tx + 1)
            wx = (x - tx * tile_w) / (tile_w + 1e-12)

            v = gray_u8[y, x]
            p11 = luts[ty][tx][v]
            p12 = luts[ty][tx2][v]
            p21 = luts[ty2][tx][v]
            p22 = luts[ty2][tx2][v]

            top = (1 - wx) * p11 + wx * p12
            bot = (1 - wx) * p21 + wx * p22
            out[y, x] = (1 - wy) * top + wy * bot

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_local_contrast_and_sharpen(img_rgb: Image.Image) -> Image.Image:
    arr = np.array(img_rgb)
    gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)

    eq = _clahe_gray_np(gray, clip_limit=CLAHE_CLIP_LIMIT, grid=CLAHE_TILE_GRID)

    ratio = (eq.astype(np.float32) + 1.0) / (gray.astype(np.float32) + 1.0)
    ratio = np.clip(ratio, 0.7, 1.6)[..., None]

    arr2 = np.clip(arr.astype(np.float32) * ratio, 0, 255).astype(np.uint8)
    img2 = Image.fromarray(arr2)

    img2 = ImageEnhance.Sharpness(img2).enhance(SHARPEN_FACTOR)
    return img2


# ======================================================
# 5) Generate input images (FULL, annotated, sub-crops) as base64 PNG
# ======================================================
def make_images_b64(img_url: str, use_enhance: bool = False):
    raw = _download_image_bytes(img_url)
    full = Image.open(BytesIO(raw)).convert("RGB")

    # Standardize size first
    full = _resize_224(full)

    # Optional enhancement (only in fallback mode)
    if use_enhance:
        full = _apply_local_contrast_and_sharpen(full)

    W, H = full.size

    # ROI box on FULL
    rx1 = int(W * ROI_X1); rx2 = int(W * ROI_X2)
    ry1 = int(H * ROI_Y1); ry2 = int(H * ROI_Y2)
    rx1, ry1, rx2, ry2 = _clamp_box(rx1, ry1, rx2, ry2, W, H)

    roi = full.crop((rx1, ry1, rx2, ry2))
    rW, rH = roi.size

    # sub-boxes inside ROI
    sub_boxes_roi = []
    for cx in HOLE_CENTERS:
        sx1 = int(rW * (cx - HOLE_BOX_W / 2))
        sx2 = int(rW * (cx + HOLE_BOX_W / 2))
        sy1 = int(rH * (HOLE_Y_CENTER - HOLE_BOX_H / 2))
        sy2 = int(rH * (HOLE_Y_CENTER + HOLE_BOX_H / 2))
        sx1, sy1, sx2, sy2 = _clamp_box(sx1, sy1, sx2, sy2, rW, rH)
        sub_boxes_roi.append((sx1, sy1, sx2, sy2))

    sub_crops = [roi.crop(b) for b in sub_boxes_roi]

    # annotated image (ROI red, Sub-ROI green)
    annotated = full.copy()
    draw = ImageDraw.Draw(annotated)

    for t in range(3):
        draw.rectangle((rx1 - t, ry1 - t, rx2 + t, ry2 + t), outline=(255, 0, 0))

    for (sx1, sy1, sx2, sy2) in sub_boxes_roi:
        ox1 = rx1 + sx1; oy1 = ry1 + sy1
        ox2 = rx1 + sx2; oy2 = ry1 + sy2
        for t in range(2):
            draw.rectangle((ox1 - t, oy1 - t, ox2 + t, oy2 + t), outline=(0, 255, 0))

    b64_full = _pil_to_b64_png(full)
    b64_ann = _pil_to_b64_png(annotated)
    b64_s1 = _pil_to_b64_png(sub_crops[0])
    b64_s2 = _pil_to_b64_png(sub_crops[1])
    b64_s3 = _pil_to_b64_png(sub_crops[2])

    meta = {
        "orig_W": W, "orig_H": H,
        "roi_box": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
        "sub_boxes_in_roi": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in sub_boxes_roi],
        "sub_params": {
            "HOLE_CENTERS": HOLE_CENTERS,
            "HOLE_Y_CENTER": HOLE_Y_CENTER,
            "HOLE_BOX_W": HOLE_BOX_W,
            "HOLE_BOX_H": HOLE_BOX_H
        },
        "preproc": {
            "resize": list(RESIZE_TO),
            "use_enhance": use_enhance,
            "clahe_clip_limit": CLAHE_CLIP_LIMIT if use_enhance else None,
            "clahe_tile_grid": list(CLAHE_TILE_GRID) if use_enhance else None,
            "sharpen_factor": SHARPEN_FACTOR if use_enhance else None,
        }
    }
    return b64_full, b64_ann, b64_s1, b64_s2, b64_s3, meta


# ======================================================
# ✅ CoVe Helpers
# ======================================================
def _has_uncertain(d: dict) -> bool:
    return any(
        str(d.get(k, "")).strip().lower() == "uncertain"
        for k in ("hole1_status", "hole2_status", "hole3_status")
    )


def _make_cove_questions(draft: dict) -> dict:
    draft_json = json.dumps(
        {
            "package_intact": draft.get("package_intact", True),
            "hole1_status": draft.get("hole1_status", "uncertain"),
            "hole2_status": draft.get("hole2_status", "uncertain"),
            "hole3_status": draft.get("hole3_status", "uncertain"),
            "confidence": draft.get("confidence", 0.0),
            "reason": draft.get("reason", "")
        },
        ensure_ascii=False
    )

    content = _post_chat([
        {"role": "system", "content": COVE_Q_SYSTEM},
        {"role": "user", "content": COVE_Q_PROMPT.format(draft_json=draft_json)},
    ])

    q = _safe_json_extract(content)
    if not isinstance(q, dict) or "questions" not in q:
        return {"questions": []}
    return q


def _cove_verify_with_images(img_url: str, questions: dict, use_enhance: bool = False) -> dict:
    b64_full, b64_ann, b64_s1, b64_s2, b64_s3, meta = make_images_b64(img_url, use_enhance=use_enhance)
    questions_json = json.dumps(questions, ensure_ascii=False)

    content = _post_chat([
        {"role": "system", "content": COVE_VERIFY_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": COVE_VERIFY_PROMPT.format(questions_json=questions_json)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_full}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ann}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s1}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s2}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s3}"}},
        ]},
    ])

    obs = _safe_json_extract(content)

    out = {
        "package_intact": bool(obs.get("package_intact", True)),
        "hole1_status": _norm_status(obs.get("hole1_status")),
        "hole2_status": _norm_status(obs.get("hole2_status")),
        "hole3_status": _norm_status(obs.get("hole3_status")),
        "confidence": float(obs.get("confidence", 0.0)),
        "reason": (str(obs.get("reason", "")).strip() or "No explanation provided"),
        "valid": True,
        "roi_meta": meta,
        "cove": {
            "used": True,
            "trigger": "uncertain_only",
            "questions": questions
        }
    }
    return out


# ======================================================
# 6) Observe once (+ if uncertain, run CoVe once)
# ======================================================
def observe_once(img_url: str, max_retries=3, use_enhance: bool = False, use_cove: bool = True) -> dict:
    for attempt in range(max_retries):
        try:
            b64_full, b64_ann, b64_s1, b64_s2, b64_s3, meta = make_images_b64(img_url, use_enhance=use_enhance)

            content = _post_chat([
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_full}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ann}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s1}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s2}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s3}"}},
                ]},
            ])

            # If the model refuses, retry with an even stricter reminder
            if _is_refusal_text(content):
                retry_prompt = (
                    "Output ONLY valid JSON. No other text.\n"
                    "Required keys: package_intact,hole1_status,hole2_status,hole3_status,confidence,reason\n\n"
                    + PROMPT
                )
                content = _post_chat([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": retry_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_full}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_ann}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s1}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s2}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_s3}"}},
                    ]},
                ])

            obs = _safe_json_extract(content)

            draft = {
                "package_intact": bool(obs.get("package_intact", True)),
                "hole1_status": _norm_status(obs.get("hole1_status")),
                "hole2_status": _norm_status(obs.get("hole2_status")),
                "hole3_status": _norm_status(obs.get("hole3_status")),
                "confidence": float(obs.get("confidence", 0.0)),
                "reason": (str(obs.get("reason", "")).strip() or "No explanation provided"),
                "valid": True,
                "roi_meta": meta,
                "cove": {"used": False},
            }

            # ✅ Run CoVe only when any hole is uncertain
            if use_cove and _has_uncertain(draft):
                questions = _make_cove_questions(draft)
                verified = _cove_verify_with_images(img_url, questions, use_enhance=use_enhance)
                return verified

            return draft

        except Exception as e:
            if attempt == max_retries - 1:
                return {
                    "package_intact": True,
                    "hole1_status": "uncertain",
                    "hole2_status": "uncertain",
                    "hole3_status": "uncertain",
                    "confidence": 0.0,
                    "reason": f"OBSERVE_FAILED: {e}",
                    "valid": False,
                    "roi_meta": {},
                    "cove": {"used": False},
                }
            time.sleep((0.5 * (2 ** attempt)) + random.uniform(0, 0.2))


# ======================================================
# 8) If conf < 0.95, do 10-run majority vote (CoVe disabled in vote runs to save cost)
# ======================================================
def observe_conf_then_vote(img_url: str, conf_thresh: float = 0.95, n_votes: int = 10, use_enhance: bool = False):
    # First run: allow CoVe (only if uncertain)
    first = observe_once(img_url, use_enhance=use_enhance, use_cove=True)

    if not first.get("valid", True):
        return first, False, {}, [first.get("reason", "")]

    if first.get("confidence", 0.0) >= conf_thresh:
        return first.copy(), False, {}, [first.get("reason", "")]

    # Vote runs: CoVe OFF (cost control)
    runs = [first] + [observe_once(img_url, use_enhance=use_enhance, use_cove=False) for _ in range(n_votes - 1)]
    runs = [r for r in runs if r.get("valid", False)]
    total = len(runs)
    if total == 0:
        return first.copy(), False, {}, [first.get("reason", "")]

    pkg_true = sum(1 for r in runs if bool(r.get("package_intact", True)))
    pkg_false = total - pkg_true
    pkg_final = (pkg_true >= pkg_false)

    def count_status(hole_idx: int, target: str) -> int:
        key = f"hole{hole_idx}_status"
        return sum(1 for r in runs if str(r.get(key, "uncertain")).strip().lower() == target)

    def majority_status(hole_idx: int) -> str:
        c = count_status(hole_idx, "connected")
        n = count_status(hole_idx, "not_connected")
        u = count_status(hole_idx, "uncertain")
        if c >= n and c >= u:
            return "connected"
        if n > c and n >= u:
            return "not_connected"
        return "uncertain"

    h1_final = majority_status(1)
    h2_final = majority_status(2)
    h3_final = majority_status(3)

    conf_avg = sum(float(r.get("confidence", 0.0)) for r in runs) / total

    vote_dist = {
        "total_runs": total,
        "pkg_true": pkg_true,
        "pkg_false": pkg_false,
        "h1": {"connected": count_status(1, "connected"), "not_connected": count_status(1, "not_connected"), "uncertain": count_status(1, "uncertain")},
        "h2": {"connected": count_status(2, "connected"), "not_connected": count_status(2, "not_connected"), "uncertain": count_status(2, "uncertain")},
        "h3": {"connected": count_status(3, "connected"), "not_connected": count_status(3, "not_connected"), "uncertain": count_status(3, "uncertain")},
        "conf_avg": conf_avg,
        "first_conf": float(first.get("confidence", 0.0)),
    }

    reason_samples = []
    for r in runs:
        s = (r.get("reason") or "").strip()
        if s and s not in reason_samples:
            reason_samples.append(s)
        if len(reason_samples) >= 3:
            break

    pkg_reason = f"Package {'intact' if pkg_final else 'damaged/suspected'} (vote {pkg_true}/{total} on FULL)."
    holes_reason = (
        f"hole1={h1_final}(c{vote_dist['h1']['connected']}/n{vote_dist['h1']['not_connected']}/u{vote_dist['h1']['uncertain']}) / "
        f"hole2={h2_final}(c{vote_dist['h2']['connected']}/n{vote_dist['h2']['not_connected']}/u{vote_dist['h2']['uncertain']}) / "
        f"hole3={h3_final}(c{vote_dist['h3']['connected']}/n{vote_dist['h3']['not_connected']}/u{vote_dist['h3']['uncertain']})."
    )
    final_reason = pkg_reason + " " + holes_reason
    if reason_samples:
        final_reason += f" | (debug samples: {' || '.join(reason_samples)})"

    final = {
        "package_intact": pkg_final,
        "hole1_status": h1_final,
        "hole2_status": h2_final,
        "hole3_status": h3_final,
        "confidence": conf_avg,
        "reason": final_reason,
        "valid": True,
        "roi_meta": first.get("roi_meta", {}),
        "cove": first.get("cove", {"used": False}),  # keep CoVe info from the first run
    }
    return final, True, vote_dist, reason_samples


# ======================================================
# 7) Label Decision Rules
# ======================================================
def decide_label(package_intact: bool, statuses: list[str]) -> int:
    if not package_intact:
        return LABEL_ABNORMAL
    s = [str(x).strip().lower() for x in statuses]
    if any(x == "not_connected" for x in s):
        return LABEL_ABNORMAL
    if all(x == "connected" for x in s):
        return LABEL_NORMAL
    return LABEL_ABNORMAL


# ======================================================
# All-0 / All-1 check
# ======================================================
def _all_same_label(labels: list[int]) -> bool:
    return len(labels) > 0 and all(l == labels[0] for l in labels)


# ======================================================
# 9) Main
# ======================================================
def main():
    if API_KEY in (None, "", "YOUR_API_KEY_HERE"):
        raise RuntimeError(
            "API key is missing. Please set LUXIA_API_KEY env var or edit API_KEY safely."
        )

    df = pd.read_csv(TEST_CSV_PATH)
    if "id" not in df.columns or "img_url" not in df.columns:
        raise ValueError(f"CSV must contain columns: id, img_url | got={df.columns.tolist()}")

    CONF_THRESH = 0.95
    N_VOTES = 10

    def run_once(use_enhance: bool):
        subs = []
        expls = []
        n = len(df)

        for i, row in df.iterrows():
            _id = row["id"]
            img_url = row["img_url"]

            final_checks, used_vote, vote_dist, _ = observe_conf_then_vote(
                img_url,
                conf_thresh=CONF_THRESH,
                n_votes=N_VOTES,
                use_enhance=use_enhance
            )

            statuses = [
                final_checks.get("hole1_status", "uncertain"),
                final_checks.get("hole2_status", "uncertain"),
                final_checks.get("hole3_status", "uncertain"),
            ]
            pkg = bool(final_checks.get("package_intact", True))

            label = decide_label(pkg, statuses)

            subs.append({"id": _id, "label": label})
            expls.append({
                "id": _id,
                "label": label,
                "confidence": round(float(final_checks.get("confidence", 0.0)), 3),
                "used_vote": used_vote,
                "valid": final_checks.get("valid", True),
                "reason": final_checks.get("reason", ""),
                "checks": json.dumps(
                    {
                        "package_intact": pkg,
                        "hole1_status": statuses[0],
                        "hole2_status": statuses[1],
                        "hole3_status": statuses[2],
                        "rule": "abnormal if package damaged OR any hole not_connected",
                        "conf_thresh": CONF_THRESH,
                        "n_votes": N_VOTES,
                        "use_enhance": use_enhance,
                        "cove_used": bool(final_checks.get("cove", {}).get("used", False)),
                        "cove_trigger": final_checks.get("cove", {}).get("trigger", ""),
                    },
                    ensure_ascii=False
                ),
                "vote_dist": json.dumps(vote_dist, ensure_ascii=False) if used_vote else "",
                "roi_meta": json.dumps(final_checks.get("roi_meta", {}), ensure_ascii=False),
            })

            tag = "ENH" if use_enhance else "RAW"
            cove_tag = "COVE" if bool(final_checks.get("cove", {}).get("used", False)) else "NO-COVE"

            print(
                f"[{tag} {cove_tag} {i+1}/{n}] id={_id} -> label={label} "
                f"| conf={final_checks.get('confidence', 0):.2f} "
                f"| pkg={'OK' if pkg else 'DAMAGED'} "
                f"| holes=({statuses[0]},{statuses[1]},{statuses[2]}) "
                f"| {final_checks.get('reason', '')}"
            )
            time.sleep(0.2)

        return subs, expls

    subs1, expls1 = run_once(use_enhance=USE_ENHANCE_DEFAULT)
    labels1 = [x["label"] for x in subs1]

    if _all_same_label(labels1):
        print("\n⚠️ All predictions are identical (all 0 or all 1).")
        print("🔁 Re-running with Local Contrast (CLAHE-like) + Sharpen...\n")
        subs2, expls2 = run_once(use_enhance=USE_ENHANCE_FALLBACK)

        pd.DataFrame(subs2, columns=["id", "label"]).to_csv(OUT_SUBMISSION, index=False)
        pd.DataFrame(expls2).to_csv(OUT_EXPLAIN, index=False)

        print(f"\n✅ Saved (fallback): {OUT_SUBMISSION}")
        print(f"✅ Saved (fallback): {OUT_EXPLAIN}")
    else:
        pd.DataFrame(subs1, columns=["id", "label"]).to_csv(OUT_SUBMISSION, index=False)
        pd.DataFrame(expls1).to_csv(OUT_EXPLAIN, index=False)

        print(f"\n✅ Saved: {OUT_SUBMISSION}")
        print(f"✅ Saved: {OUT_EXPLAIN}")


if __name__ == "__main__":
    main()