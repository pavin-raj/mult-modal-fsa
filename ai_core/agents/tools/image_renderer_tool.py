"""
Image Renderer Tool — Option B (Annotated Input Image).

Draws the VLM's bounding boxes + labels directly onto the technician's photo,
so the response is visual ("here is exactly where the leak is") instead of text-only.

Robust to bbox coordinate format: auto-detects absolute pixels, 0-1, 0-100, or 0-1000.
"""
import os
import io
import time
import base64
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import structlog

logger = structlog.get_logger(__name__)

# Keyword -> color mapping for fault severity (extend freely)
FAULT_COLORS = {
    "leak": "#FF3B30", "seal": "#FF3B30", "crack": "#FF3B30", "fracture": "#FF3B30",
    "corrosion": "#FF9500", "rust": "#FF9500", "bearing": "#FF9500", "overheat": "#FF9500",
    "wear": "#FFCC00", "vibration": "#FFCC00", "misalign": "#FFCC00",
    "coupling": "#007AFF", "motor": "#5856D6", "wiring": "#5856D6",
}
PALETTE = ["#34C759", "#00C7BE", "#32ADE6", "#AF52DE", "#FF2D55", "#A2845E"]


def _decode_image(image_b64: str):
    from PIL import Image
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _encode_image(img, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _normalize_bbox(bbox: List[float], W: int, H: int):
    """Convert any common bbox format to absolute pixel coords, clamped to image."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    max_val = max(abs(x1), abs(y1), abs(x2), abs(y2))

    if max_val <= 1.0:                                   # 0-1 normalized
        x1, x2, y1, y2 = x1 * W, x2 * W, y1 * H, y2 * H
    elif max_val <= 100.0:                               # 0-100
        x1, x2, y1, y2 = x1 / 100 * W, x2 / 100 * W, y1 / 100 * H, y2 / 100 * H
    elif max_val <= 1000.0:                              # 0-1000 (Gemini-style)
        x1, x2, y1, y2 = x1 / 1000 * W, x2 / 1000 * W, y1 / 1000 * H, y2 / 1000 * H
    # else: already absolute pixels

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return (max(0, min(W, x1)), max(0, min(H, y1)),
            max(0, min(W, x2)), max(0, min(H, y2)))


def _color_for_label(label: str) -> str:
    l = label.lower()
    for kw, color in FAULT_COLORS.items():
        if kw in l:
            return color
    return PALETTE[hash(label) % len(PALETTE)]


def _text_color_for(bg_hex: str) -> str:
    """Pick black/white text for contrast against the box color."""
    h = bg_hex.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#000000" if luminance > 0.6 else "#FFFFFF"


def _load_font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


@tool
def annotate_image(
    image_b64: str,
    bounding_boxes: List[Dict[str, Any]],
    detected_faults: Optional[List[str]] = None,
    show_legend: bool = True,
) -> dict:
    """
    Draw detected-fault bounding boxes and labels onto a field photo.

    Use this whenever vision analysis produced bounding boxes and you want to
    return a VISUAL result (annotated image) to the technician.
    """
    start = time.time()
    from PIL import ImageDraw

    try:
        img = _decode_image(image_b64)
    except Exception as e:
        return {"success": False, "data": None, "error": f"decode failed: {e}",
                "tool_name": "annotate_image", "latency_ms": 0}

    W, H = img.size
    draw = ImageDraw.Draw(img)
    line_w = max(3, int(min(W, H) * 0.004))
    font = _load_font(max(14, int(min(W, H) * 0.022)))

    boxes_drawn = 0
    labels_used: List[str] = []
    legend: Dict[str, str] = {}   # label -> color (first-seen order)

    for box in bounding_boxes or []:
        raw = box.get("bbox") or [box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2")]
        if not raw or any(v is None for v in raw[:4]):
            continue
        x1, y1, x2, y2 = _normalize_bbox(raw, W, H)
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            continue  # degenerate box

        label = (box.get("label") or "fault").replace("_", " ").title()
        color = _color_for_label(label)
        legend.setdefault(label, color)

        # rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        # label background (above box if room, else just inside top edge)
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad = 4
        ly1 = y1 - th - pad * 2 if y1 - th - pad * 2 >= 0 else y1
        draw.rectangle([x1, ly1, x1 + tw + pad * 2, ly1 + th + pad * 2], fill=color)
        draw.text((x1 + pad, ly1 + pad), label, fill=_text_color_for(color), font=font)

        boxes_drawn += 1
        labels_used.append(label)

    # ---- legend (top-left panel) ----
    if show_legend and legend:
        from PIL import Image
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        sw = max(14, int(min(W, H) * 0.02))
        row_h = sw + 8
        panel_w = max(int(W * 0.32), 160)
        panel_h = row_h * len(legend) + 16
        od.rectangle([8, 8, 8 + panel_w, 8 + panel_h],
                     fill=(255, 255, 255, 225), outline=(0, 0, 0, 200), width=1)
        for i, (lbl, col) in enumerate(legend.items()):
            ry = 14 + i * row_h
            od.rectangle([16, ry, 16 + sw, ry + sw], fill=col)
            od.text((16 + sw + 8, ry - 2), lbl, fill=(0, 0, 0, 255), font=font)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    caption = (f"Detected: {', '.join(legend.keys())}" if legend else "No faults localized")

    latency = int((time.time() - start) * 1000)
    logger.info("annotate_image.done", boxes=boxes_drawn, size=f"{W}x{H}", latency_ms=latency)

    return {
        "success": True,
        "data": {
            "image_b64": _encode_image(img, "PNG"),
            "format": "png",
            "width": W,
            "height": H,
            "boxes_drawn": boxes_drawn,
            "labels": list(legend.keys()),
            "caption": caption,
        },
        "error": None,
        "tool_name": "annotate_image",
        "latency_ms": latency,
    }