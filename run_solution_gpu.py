import os
import re
import json
import time
import base64
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
import torch

# 스크립트 위치 기준으로 작업 디렉토리 설정
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)

# ======================================================
# 0) GPU / Model Configuration
# ======================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("WARNING: CUDA not available, running on CPU (slow)")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# ======================================================
# 1) Load Model (once at startup)
# ======================================================
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

print(f"\nLoading {MODEL_NAME} with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
print("Model loaded successfully!\n")

# ======================================================
# 2) I/O Paths
# ======================================================
TEST_CSV_PATH = "./test.csv"
OUT_SUBMISSION = "./submission.csv"
OUT_EXPLAIN = "./explanations.csv"

LABEL_NORMAL = 0
LABEL_ABNORMAL = 1

# ======================================================
# 3) Image Preprocessing Config
# ======================================================
RESIZE_TO = (512, 512)  # 224→512: 세부 검사 정확도 향상
USE_ENHANCE_DEFAULT = False
USE_ENHANCE_FALLBACK = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
SHARPEN_FACTOR = 1.8

# ROI / Sub-ROI Settings
ROI_X1 = 0.18
ROI_X2 = 0.82
ROI_Y1 = 0.45
ROI_Y2 = 0.98
HOLE_CENTERS = [0.18, 0.50, 0.82]
HOLE_Y_CENTER = 0.78
HOLE_BOX_W = 0.30
HOLE_BOX_H = 0.45

# ======================================================
# 4) Prompts
# ======================================================
SYSTEM = (
    "You are an inspector for semiconductor device visual inspection in a manufacturing process.\n"
    "You MUST output only a valid JSON object. Do NOT output any other sentences, explanations, or code blocks.\n"
)

PROMPT = r"""Input images (5):
1) FULL: the original full image (package damage MUST be judged using ONLY this image)
2) ANNOTATED: the original image with ROI (red) + three Sub-ROIs (green) drawn (for location reference)
3) Sub-ROI #1: crop around the LEFT hole (for lead connection judgment)
4) Sub-ROI #2: crop around the CENTER hole (for lead connection judgment)
5) Sub-ROI #3: crop around the RIGHT hole (for lead connection judgment)

[What you must judge]
A) package_intact:
- From FULL only, return true if there is NO chipping, cracking, breakage, or damaged corners/edges on the package (black body).
- Return false if any damage is visible.

B) hole1_status, hole2_status, hole3_status:
- connected: The metal lead directly touches the hole at the TOP (12 o'clock), descends into the hole with NO visible vertical gap.
- not_connected: Lead not at top of hole, OR clear empty space (gap) between lead and hole.
- uncertain: Cannot determine 12 o'clock contact or gap presence.

[Reason] Must include: (1) Package evidence from FULL. (2) Each hole: 12 o'clock contact + gap check.

[Output ONLY this JSON, no other text]
{"package_intact": true, "hole1_status": "connected", "hole2_status": "connected", "hole3_status": "connected", "confidence": 0.0, "reason": ""}"""

COVE_Q_SYSTEM = (
    "You are a verification-question generator for manufacturing visual inspection.\n"
    "Output valid JSON only.\n"
)

COVE_Q_PROMPT = r"""Create verification questions for the draft decision below.
For each hole: check (1) 12 o'clock contact, (2) gap presence.
For package_intact: check FULL image only.

[Output JSON]
{"questions": [{"key": "package_intact", "q": "..."}, {"key": "hole1_status", "q": "..."}, {"key": "hole2_status", "q": "..."}, {"key": "hole3_status", "q": "..."}]}

[draft]
{draft_json}"""

COVE_VERIFY_SYSTEM = (
    "You are a semiconductor inspector in the verification stage.\n"
    "Output only a valid JSON object. Re-judge independently from the draft.\n"
)

COVE_VERIFY_PROMPT = r"""Images: FULL, ANNOTATED, Sub-ROI #1, #2, #3.
Answer checklist questions then output ONLY the final JSON.

[Checklist]
{questions_json}

[Output ONLY this JSON]
{"package_intact": true, "hole1_status": "connected", "hole2_status": "connected", "hole3_status": "connected", "confidence": 0.0, "reason": ""}"""


# ======================================================
# 5) Local GPU Inference
# ======================================================
def _b64_to_pil(data_url: str) -> Image.Image:
    """base64 data URL → PIL Image (RGB)"""
    # data:image/png;base64,<data>
    _, b64data = data_url.split(",", 1)
    return Image.open(BytesIO(base64.b64decode(b64data))).convert("RGB")


def _post_chat_local(messages: list) -> str:
    """Local GPU inference with Qwen2.5-VL.
    base64 data URLs → PIL Image 변환 후 processor에 직접 전달.
    """
    # Convert OpenAI-style messages → Qwen2.5-VL format
    # image_url 항목은 PIL Image로 미리 디코딩해서 넘김
    qwen_messages = []
    pil_images = []  # 순서대로 수집

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            qwen_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            qwen_content = []
            for item in content:
                if item["type"] == "text":
                    qwen_content.append({"type": "text", "text": item["text"]})
                elif item["type"] == "image_url":
                    url = item["image_url"]["url"]
                    pil_img = _b64_to_pil(url)
                    pil_images.append(pil_img)
                    # Qwen2.5-VL 형식: image placeholder
                    qwen_content.append({"type": "image", "image": pil_img})
            qwen_messages.append({"role": role, "content": qwen_content})

    text = processor.apply_chat_template(
        qwen_messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=pil_images if pil_images else None,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    raw = output_text[0].strip()
    return raw


# ======================================================
# 6) Utilities
# ======================================================
def _is_refusal_text(s: str) -> bool:
    s_low = s.strip().lower()
    return ("i'm sorry" in s_low and "can't assist" in s_low) or ("i can't assist" in s_low)


def _safe_json_extract(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        # ```json ... ``` 블록 제거 후 재시도
        cleaned = re.sub(r"```(?:json)?\s*", "", s).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\{.*\}", cleaned, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    print(f"[DEBUG] JSON parse failed. Raw output: {repr(s[:300])}")
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


def _resize_224(img: Image.Image) -> Image.Image:
    return img.resize(RESIZE_TO, resample=Image.BICUBIC)


def _clahe_gray_np(gray_u8: np.ndarray, clip_limit: float = 2.0, grid=(8, 8)) -> np.ndarray:
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
# 7) Generate ROI images
# ======================================================
def make_images_b64(img_url: str, use_enhance: bool = False):
    raw = _download_image_bytes(img_url)
    full = Image.open(BytesIO(raw)).convert("RGB")
    full = _resize_224(full)
    if use_enhance:
        full = _apply_local_contrast_and_sharpen(full)

    W, H = full.size
    rx1 = int(W * ROI_X1); rx2 = int(W * ROI_X2)
    ry1 = int(H * ROI_Y1); ry2 = int(H * ROI_Y2)
    rx1, ry1, rx2, ry2 = _clamp_box(rx1, ry1, rx2, ry2, W, H)
    roi = full.crop((rx1, ry1, rx2, ry2))
    rW, rH = roi.size

    sub_boxes_roi = []
    for cx in HOLE_CENTERS:
        sx1 = int(rW * (cx - HOLE_BOX_W / 2))
        sx2 = int(rW * (cx + HOLE_BOX_W / 2))
        sy1 = int(rH * (HOLE_Y_CENTER - HOLE_BOX_H / 2))
        sy2 = int(rH * (HOLE_Y_CENTER + HOLE_BOX_H / 2))
        sx1, sy1, sx2, sy2 = _clamp_box(sx1, sy1, sx2, sy2, rW, rH)
        sub_boxes_roi.append((sx1, sy1, sx2, sy2))

    sub_crops = [roi.crop(b) for b in sub_boxes_roi]

    annotated = full.copy()
    draw = ImageDraw.Draw(annotated)
    for t in range(3):
        draw.rectangle((rx1 - t, ry1 - t, rx2 + t, ry2 + t), outline=(255, 0, 0))
    for (sx1, sy1, sx2, sy2) in sub_boxes_roi:
        ox1 = rx1 + sx1; oy1 = ry1 + sy1
        ox2 = rx1 + sx2; oy2 = ry1 + sy2
        for t in range(2):
            draw.rectangle((ox1 - t, oy1 - t, ox2 + t, oy2 + t), outline=(0, 255, 0))

    meta = {
        "orig_W": W, "orig_H": H,
        "roi_box": {"x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2},
        "sub_boxes_in_roi": [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]} for b in sub_boxes_roi],
    }
    return (
        _pil_to_b64_png(full),
        _pil_to_b64_png(annotated),
        _pil_to_b64_png(sub_crops[0]),
        _pil_to_b64_png(sub_crops[1]),
        _pil_to_b64_png(sub_crops[2]),
        meta,
    )


# ======================================================
# 8) CoVe Helpers
# ======================================================
def _has_uncertain(d: dict) -> bool:
    return any(
        str(d.get(k, "")).strip().lower() == "uncertain"
        for k in ("hole1_status", "hole2_status", "hole3_status")
    )


def _make_cove_questions(draft: dict) -> dict:
    draft_json = json.dumps({
        "package_intact": draft.get("package_intact", True),
        "hole1_status": draft.get("hole1_status", "uncertain"),
        "hole2_status": draft.get("hole2_status", "uncertain"),
        "hole3_status": draft.get("hole3_status", "uncertain"),
        "confidence": draft.get("confidence", 0.0),
        "reason": draft.get("reason", ""),
    }, ensure_ascii=False)
    content = _post_chat_local([
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
    content = _post_chat_local([
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
    return {
        "package_intact": bool(obs.get("package_intact", True)),
        "hole1_status": _norm_status(obs.get("hole1_status")),
        "hole2_status": _norm_status(obs.get("hole2_status")),
        "hole3_status": _norm_status(obs.get("hole3_status")),
        "confidence": float(obs.get("confidence", 0.0)),
        "reason": str(obs.get("reason", "")).strip() or "No explanation provided",
        "valid": True,
        "roi_meta": meta,
        "cove": {"used": True, "trigger": "uncertain_only", "questions": questions},
    }


# ======================================================
# 9) Observe once (+ CoVe if uncertain)
# ======================================================
def observe_once(img_url: str, max_retries=3, use_enhance: bool = False, use_cove: bool = True) -> dict:
    for attempt in range(max_retries):
        try:
            b64_full, b64_ann, b64_s1, b64_s2, b64_s3, meta = make_images_b64(img_url, use_enhance=use_enhance)
            content = _post_chat_local([
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
            if _is_refusal_text(content):
                continue

            # 첫 번째 호출 시 원본 출력 확인
            if attempt == 0 and not getattr(observe_once, '_debug_shown', False):
                print(f"\n[DEBUG] Model raw output (first call):\n{repr(content[:400])}\n")
                observe_once._debug_shown = True

            obs = _safe_json_extract(content)
            draft = {
                "package_intact": bool(obs.get("package_intact", True)),
                "hole1_status": _norm_status(obs.get("hole1_status")),
                "hole2_status": _norm_status(obs.get("hole2_status")),
                "hole3_status": _norm_status(obs.get("hole3_status")),
                "confidence": float(obs.get("confidence", 0.0)),
                "reason": str(obs.get("reason", "")).strip() or "No explanation provided",
                "valid": True,
                "roi_meta": meta,
                "cove": {"used": False},
            }
            if use_cove and _has_uncertain(draft):
                questions = _make_cove_questions(draft)
                return _cove_verify_with_images(img_url, questions, use_enhance=use_enhance)
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
            time.sleep(0.3 * (attempt + 1))


# ======================================================
# 10) Confidence-based voting (3 votes, GPU-optimized)
# ======================================================
def observe_conf_then_vote(img_url: str, conf_thresh=0.85, n_votes=3, use_enhance=False):
    first = observe_once(img_url, use_enhance=use_enhance, use_cove=True)
    if not first.get("valid", True):
        return first, False, {}, []
    if first.get("confidence", 0.0) >= conf_thresh:
        return first.copy(), False, {}, []

    runs = [first] + [observe_once(img_url, use_enhance=use_enhance, use_cove=False) for _ in range(n_votes - 1)]
    runs = [r for r in runs if r.get("valid", False)]
    total = len(runs)
    if total == 0:
        return first.copy(), False, {}, []

    pkg_true = sum(1 for r in runs if bool(r.get("package_intact", True)))
    pkg_final = pkg_true >= (total - pkg_true)

    def count_status(idx, target):
        return sum(1 for r in runs if str(r.get(f"hole{idx}_status", "uncertain")).lower() == target)

    def majority_status(idx):
        c, n, u = count_status(idx, "connected"), count_status(idx, "not_connected"), count_status(idx, "uncertain")
        if c >= n and c >= u:
            return "connected"
        if n > c and n >= u:
            return "not_connected"
        return "uncertain"

    h1, h2, h3 = majority_status(1), majority_status(2), majority_status(3)
    conf_avg = sum(float(r.get("confidence", 0.0)) for r in runs) / total

    vote_dist = {
        "total_runs": total, "pkg_true": pkg_true,
        "h1": {s: count_status(1, s) for s in ["connected", "not_connected", "uncertain"]},
        "h2": {s: count_status(2, s) for s in ["connected", "not_connected", "uncertain"]},
        "h3": {s: count_status(3, s) for s in ["connected", "not_connected", "uncertain"]},
        "conf_avg": conf_avg,
    }
    final = {
        "package_intact": pkg_final,
        "hole1_status": h1, "hole2_status": h2, "hole3_status": h3,
        "confidence": conf_avg,
        "reason": f"Vote({total}): pkg={pkg_true}/{total} h1={h1} h2={h2} h3={h3}",
        "valid": True,
        "roi_meta": first.get("roi_meta", {}),
        "cove": first.get("cove", {"used": False}),
    }
    return final, True, vote_dist, []


# ======================================================
# 11) Label Decision
# ======================================================
def decide_label(package_intact: bool, statuses: list) -> int:
    if not package_intact:
        return LABEL_ABNORMAL
    s = [str(x).strip().lower() for x in statuses]
    if any(x == "not_connected" for x in s):
        return LABEL_ABNORMAL
    if all(x == "connected" for x in s):
        return LABEL_NORMAL
    return LABEL_ABNORMAL


def _all_same_label(labels: list) -> bool:
    return len(labels) > 0 and all(l == labels[0] for l in labels)


# ======================================================
# 12) Main
# ======================================================
def main():
    df = pd.read_csv(TEST_CSV_PATH)
    assert "id" in df.columns and "img_url" in df.columns, f"Need id,img_url columns. Got: {df.columns.tolist()}"
    print(f"Loaded {len(df)} samples from {TEST_CSV_PATH}\n")

    CONF_THRESH = 0.85
    N_VOTES = 3  # Reduced from 10: local inference is deterministic (greedy)

    def run_once(use_enhance: bool):
        subs, expls = [], []
        n = len(df)
        for i, row in df.iterrows():
            _id, img_url = row["id"], row["img_url"]
            final, used_vote, vote_dist, _ = observe_conf_then_vote(
                img_url, conf_thresh=CONF_THRESH, n_votes=N_VOTES, use_enhance=use_enhance
            )
            statuses = [final.get(f"hole{j}_status", "uncertain") for j in (1, 2, 3)]
            pkg = bool(final.get("package_intact", True))
            label = decide_label(pkg, statuses)

            subs.append({"id": _id, "label": label})
            expls.append({
                "id": _id, "label": label,
                "confidence": round(float(final.get("confidence", 0.0)), 3),
                "used_vote": used_vote,
                "valid": final.get("valid", True),
                "reason": final.get("reason", ""),
                "checks": json.dumps({
                    "package_intact": pkg,
                    "hole1_status": statuses[0], "hole2_status": statuses[1], "hole3_status": statuses[2],
                    "cove_used": bool(final.get("cove", {}).get("used", False)),
                }, ensure_ascii=False),
                "vote_dist": json.dumps(vote_dist, ensure_ascii=False) if used_vote else "",
                "roi_meta": json.dumps(final.get("roi_meta", {}), ensure_ascii=False),
            })

            tag = "ENH" if use_enhance else "RAW"
            print(
                f"[GPU {tag} {i+1}/{n}] {_id} -> label={label} "
                f"conf={final.get('confidence',0):.2f} "
                f"pkg={'OK' if pkg else 'NG'} "
                f"holes=({statuses[0]},{statuses[1]},{statuses[2]})"
            )
        return subs, expls

    subs1, expls1 = run_once(use_enhance=False)
    labels1 = [x["label"] for x in subs1]

    if _all_same_label(labels1):
        print(f"\nAll labels identical ({labels1[0]}). Re-running with CLAHE+Sharpen fallback...")
        subs2, expls2 = run_once(use_enhance=True)
        pd.DataFrame(subs2, columns=["id", "label"]).to_csv(OUT_SUBMISSION, index=False)
        pd.DataFrame(expls2).to_csv(OUT_EXPLAIN, index=False)
        print(f"\nSaved (fallback): {OUT_SUBMISSION}")
    else:
        pd.DataFrame(subs1, columns=["id", "label"]).to_csv(OUT_SUBMISSION, index=False)
        pd.DataFrame(expls1).to_csv(OUT_EXPLAIN, index=False)
        print(f"\nSaved: {OUT_SUBMISSION}")
        label_counts = pd.Series(labels1).value_counts().to_dict()
        print(f"Label distribution: {label_counts}")


if __name__ == "__main__":
    main()
