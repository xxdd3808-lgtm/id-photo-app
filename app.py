"""
证件照智能生成器 - 本地 Web 应用
依赖: streamlit, rembg, Pillow, opencv-python-headless, numpy
"""

import io
import math
import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image, ImageEnhance, ImageDraw
from rembg import remove, new_session

# ─────────────────────────────────────────────
# 常量定义
# ─────────────────────────────────────────────

SIZES = {
    "标准1寸 (25×35mm)": (295, 413),
    "标准2寸 (35×49mm)": (413, 579),
    "小2寸/护照 (33×48mm)": (390, 567),
}

DPI_OPTIONS = {
    "标准 (300 DPI)": 1,
    "高清 (600 DPI)": 2,
}

def get_target_size(size_name: str, dpi_label: str) -> tuple:
    w, h = SIZES[size_name]
    mul = DPI_OPTIONS[dpi_label]
    return w * mul, h * mul

BG_COLORS = {
    "白底": "#FFFFFF",
    "蓝底": "#438EDB",
    "红底": "#C8102E",
}

CANVAS_W, CANVAS_H = 1200, 1800
GUIDE_LINE_WIDTH = 4
GUIDE_LINE_COLOR = (200, 200, 200)

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


MODEL_NAME = "u2net_human_seg"
MAX_INPUT_PX = 1600  # 中等分辨率，兼顾质量和内存

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _limit_image_size(img: Image.Image, max_px: int = MAX_INPUT_PX) -> Image.Image:
    """如果图片最长边超过 max_px，等比缩放。"""
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    ratio = max_px / max(w, h)
    new_size = (int(w * ratio), int(h * ratio))
    print(f"[resize] {w}x{h} → {new_size[0]}x{new_size[1]}", flush=True)
    return img.resize(new_size, Image.LANCZOS)


def _refine_alpha(rgb: np.ndarray, alpha: np.ndarray, radius: int = 10, eps: float = 1e-5) -> np.ndarray:
    """Guided Filter 精修 alpha 遮罩 — 用原图结构引导，锐化发丝边缘。"""
    guide = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    src = alpha.astype(np.float32)
    ksize = (radius, radius)

    mean_I = cv2.boxFilter(guide, -1, ksize)
    mean_p = cv2.boxFilter(src, -1, ksize)
    mean_Ip = cv2.boxFilter(guide * src, -1, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, -1, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)

    refined = mean_a * guide + mean_b
    return np.clip(refined, 0, 255).astype(np.uint8)


@st.cache_resource(show_spinner="正在加载 AI 模型（首次需下载约 180MB）…")
def _load_rembg_session():
    print(f"[rembg] Loading {MODEL_NAME}...", flush=True)
    session = new_session(MODEL_NAME)
    print(f"[rembg] Model loaded OK", flush=True)
    import gc
    gc.collect()
    return session


def remove_background(img: Image.Image) -> Image.Image:
    try:
        print(f"[remove_bg] Start, image size={img.size}", flush=True)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        session = _load_rembg_session()
        print(f"[remove_bg] Running inference…", flush=True)
        result = remove(buf.read(), session=session)
        print(f"[remove_bg] Done, output={len(result)} bytes", flush=True)

        fg = Image.open(io.BytesIO(result)).convert("RGBA")
        arr = np.array(fg)
        # Guided filter 精修 alpha，大幅改善发丝边缘
        refined_alpha = _refine_alpha(arr[:, :, :3], arr[:, :, 3])
        arr[:, :, 3] = refined_alpha
        return Image.fromarray(arr, "RGBA")
    except Exception as e:
        print(f"[remove_bg] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        st.error(f"抠图失败: {str(e)}")
        st.code(traceback.format_exc())
        return None


def apply_background(fg: Image.Image, bg_hex: str) -> Image.Image:
    """背景合成：Alpha 羽化后直接 alpha 混合。"""
    rgb = hex_to_rgb(bg_hex)
    rgba = np.array(fg).astype(np.float32)
    rgb_data = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # Alpha 羽化：让边缘过渡更自然
    alpha_u8 = np.clip(alpha, 0, 255).astype(np.uint8)
    alpha_smooth = cv2.GaussianBlur(alpha_u8, (3, 3), sigmaX=1.0)
    alpha_norm = np.clip(alpha_smooth.astype(np.float32) / 255.0, 0, 1)

    # Alpha 混合
    bg = np.array(rgb, dtype=np.float32)
    result = np.empty_like(rgb_data)
    for c in range(3):
        result[:, :, c] = rgb_data[:, :, c] * alpha_norm + bg[c] * (1 - alpha_norm)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8), "RGB")


def _detect_face_rect(img: Image.Image):
    """返回最大人脸的像素矩形 (x, y, w, h)，未检测到返回 None。"""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        return None
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def _calc_crop_rect(img_w: int, img_h: int, target_w: int, target_h: int,
                    face, offset_h: float = 0.0, offset_v: float = 0.0):
    """
    计算证件照裁剪矩形（在原图坐标系中）。
    返回 (left, top, crop_w, crop_h, scale)。
    offset_h: 水平偏移比 (-1~1, 0=自动居中)
    offset_v: 垂直偏移比 (-1~1, 负=上移, 正=下移)
    """
    if face is None:
        target_aspect = target_w / target_h
        src_aspect = img_w / img_h
        if src_aspect > target_aspect:
            crop_h = img_h
            crop_w = int(img_h * target_aspect)
        else:
            crop_w = img_w
            crop_h = int(img_w / target_aspect)
        left = (img_w - crop_w) // 2
        top = max(0, int((img_h - crop_h) * 0.25))
        top = min(top, img_h - crop_h)
        return (left, top, crop_w, crop_h, crop_w / target_w)

    fx, fy, fw, fh = face
    HEAD_EXTEND = 0.45
    head_top_y = fy - fh * HEAD_EXTEND
    chin_y = fy + fh
    full_head_h = chin_y - head_top_y

    face_ratio = 0.65
    top_margin_ratio = 0.05

    scale = (face_ratio * target_h) / full_head_h
    crop_w_orig = target_w / scale
    crop_h_orig = target_h / scale

    face_cx_orig = fx + fw / 2
    crop_left = int(face_cx_orig - crop_w_orig / 2 + offset_h * crop_w_orig * 0.25)
    crop_top = int(head_top_y - top_margin_ratio * crop_h_orig + offset_v * crop_h_orig * 0.20)

    crop_left = max(0, min(crop_left, img_w - math.ceil(crop_w_orig)))
    crop_top = max(0, min(crop_top, img_h - math.ceil(crop_h_orig)))
    crop_w_orig = min(math.ceil(crop_w_orig), img_w - crop_left)
    crop_h_orig = min(math.ceil(crop_h_orig), img_h - crop_top)

    return (crop_left, crop_top, crop_w_orig, crop_h_orig, scale)


def smart_crop(img: Image.Image, target_w: int, target_h: int,
               face=None, offset_h: float = 0.0, offset_v: float = 0.0) -> Image.Image:
    """
    从原图裁剪 ROI 后缩放到目标尺寸（高画质路径）。
    """
    if face is None:
        face = _detect_face_rect(img)
    left, top, cw, ch, _ = _calc_crop_rect(
        img.width, img.height, target_w, target_h, face, offset_h, offset_v
    )
    roi = img.crop((left, top, left + cw, top + ch))
    return roi.resize((target_w, target_h), Image.LANCZOS)


def draw_crop_overlay(img: Image.Image, target_w: int, target_h: int,
                      face, offset_h: float = 0.0, offset_v: float = 0.0) -> Image.Image:
    """在全图上绘制裁剪框覆盖层，用于预览。"""
    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)
    left, top, cw, ch, _ = _calc_crop_rect(
        img.width, img.height, target_w, target_h, face, offset_h, offset_v
    )
    # 半透明遮罩（裁剪区域外）
    mask = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rectangle([(0, 0), (overlay.width, top)], fill=(0, 0, 0, 100))
    mask_draw.rectangle([(0, top + ch), (overlay.width, overlay.height)], fill=(0, 0, 0, 100))
    mask_draw.rectangle([(0, top), (left, top + ch)], fill=(0, 0, 0, 100))
    mask_draw.rectangle([(left + cw, top), (overlay.width, top + ch)], fill=(0, 0, 0, 100))
    # 裁剪框边线（白色虚线效果 — 实线替代）
    draw.rectangle([(left, top), (left + cw, top + ch)], outline=(0, 255, 100), width=3)
    # 人脸检测框（蓝色）
    if face is not None:
        fx, fy, fw, fh = face
        draw.rectangle([(fx, fy), (fx + fw, fy + fh)], outline=(0, 120, 255), width=2)

    overlay = Image.alpha_composite(overlay, mask)
    return overlay.convert("RGB")


def adjust_image(img: Image.Image, brightness: float, contrast: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


# ─────────────────────────────────────────────
# 美颜处理
# ─────────────────────────────────────────────

def _skin_mask(img_array: np.ndarray) -> np.ndarray:
    """HSV 色彩空间提取皮肤区域，返回 0~1 浮点掩码。"""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # 肤色范围（HSV）：H 0~25 或 160~180, S 30~200, V 60~255
    mask1 = (h <= 25) & (s >= 30) & (s <= 200) & (v >= 60)
    mask2 = (h >= 160) & (s >= 30) & (s <= 200) & (v >= 60)
    mask = (mask1 | mask2).astype(np.float32)
    # 高斯模糊使掩码边缘平滑
    mask = cv2.GaussianBlur(mask, (15, 15), 5.0)
    return np.clip(mask, 0, 1)


def _smooth_skin(img_array: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    """双边滤波磨皮，仅作用于皮肤区域。"""
    if strength <= 0:
        return img_array
    # 根据强度调整双边滤波参数
    d = 9
    sigma_color = 50 + strength * 50   # 50~100
    sigma_space = 50 + strength * 50
    smoothed = cv2.bilateralFilter(img_array, d, sigma_color, sigma_space)
    # 用掩码混合原图和磨皮结果
    mask_3c = np.stack([mask] * 3, axis=-1)
    result = img_array.astype(np.float32) * (1 - mask_3c * strength) + \
             smoothed.astype(np.float32) * mask_3c * strength
    return np.clip(result, 0, 255).astype(np.uint8)


def _brighten_face(img_array: np.ndarray, face_rect, strength: float) -> np.ndarray:
    """对面部区域做局部提亮（gamma 校正 + 椭圆高斯蒙版）。"""
    if strength <= 0 or face_rect is None:
        return img_array
    fx, fy, fw, fh = face_rect
    h, w = img_array.shape[:2]
    # 扩展人脸区域
    ext = 0.3
    x1 = max(0, int(fx - fw * ext))
    y1 = max(0, int(fy - fh * ext))
    x2 = min(w, int(fx + fw * (1 + ext)))
    y2 = min(h, int(fy + fh * (1 + ext)))

    # 创建椭圆高斯蒙版
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    ax = (x2 - x1) / 2
    ay = (y2 - y1) / 2
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    gaussian = np.exp(-(((xx - cx) / max(ax, 1)) ** 2 + ((yy - cy) / max(ay, 1)) ** 2))
    gaussian = gaussian * strength  # 0~strength

    # Gamma 校正提亮：gamma < 1 提亮
    gamma = 1.0 - strength * 0.3  # 1.0~0.7
    roi = img_array[y1:y2, x1:x2].astype(np.float32) / 255.0
    roi_bright = np.power(roi, gamma) * 255.0
    roi_bright = np.clip(roi_bright, 0, 255)

    # 高斯蒙版混合
    mask_3c = np.stack([gaussian] * 3, axis=-1)
    blended = img_array[y1:y2, x1:x2].astype(np.float32) * (1 - mask_3c) + \
              roi_bright * mask_3c
    result = img_array.copy()
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return result


def beautify_image(img: Image.Image, smoothing: float, face_brighten: float,
                   saturation: float, sharpness: float,
                   face_rect=None) -> Image.Image:
    """美颜处理：磨皮 + 面部提亮 + 饱和度 + 锐度。"""
    arr = np.array(img)

    # 磨皮（需要皮肤掩码）
    if smoothing > 0:
        mask = _skin_mask(arr)
        arr = _smooth_skin(arr, mask, smoothing)

    # 面部提亮
    if face_brighten > 0 and face_rect is not None:
        arr = _brighten_face(arr, face_rect, face_brighten)

    result = Image.fromarray(arr, "RGB")

    # 饱和度
    if saturation != 1.0:
        result = ImageEnhance.Color(result).enhance(saturation)

    # 锐度
    if sharpness != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(sharpness)

    return result


def auto_enhance(img: Image.Image) -> tuple:
    """一键自动优化：白平衡 + CLAHE 曝光 + 饱和度。返回 (处理后图片, 建议参数dict)。"""
    arr = np.array(img).astype(np.float32)

    # 1. Gray World 自动白平衡（加阻尼，避免过度校正）
    avg_r, avg_g, avg_b = arr[:, :, 0].mean(), arr[:, :, 1].mean(), arr[:, :, 2].mean()
    avg_all = (avg_r + avg_g + avg_b) / 3
    damping = 0.5  # 只校正一半，避免过度偏移
    if avg_r > 0 and avg_g > 0 and avg_b > 0:
        arr[:, :, 0] = arr[:, :, 0] * (1 + (avg_all / avg_r - 1) * damping)
        arr[:, :, 1] = arr[:, :, 1] * (1 + (avg_all / avg_g - 1) * damping)
        arr[:, :, 2] = arr[:, :, 2] * (1 + (avg_all / avg_b - 1) * damping)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 2. CLAHE 自适应曝光（LAB 色彩空间对 L 通道操作，保守参数）
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 3. 轻微饱和度提升
    result = Image.fromarray(arr, "RGB")
    result = ImageEnhance.Color(result).enhance(1.05)

    # 返回图片和建议参数
    suggestions = {
        "smoothing": 0.2,
        "face_brighten": 0.05,
        "saturation": 1.05,
        "sharpness": 1.0,
    }
    return result, suggestions


def build_layout_image(photo: Image.Image, size_name: str) -> Image.Image:
    pw, ph = photo.size

    if "1寸" in size_name and "2寸" not in size_name:
        cols, rows = 3, 3
    else:
        cols, rows = 2, 2

    # 间距固定为照片宽高的 15%
    gap_x = max(10, int(pw * 0.15))
    gap_y = max(10, int(ph * 0.15))

    # 画布尺寸根据照片实际大小动态计算
    canvas_w = cols * pw + (cols + 1) * gap_x
    canvas_h = rows * ph + (rows + 1) * gap_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    total_photo_w = cols * pw
    total_photo_h = rows * ph

    positions = []
    for r in range(rows):
        for c in range(cols):
            x = gap_x + c * (pw + gap_x)
            y = gap_y + r * (ph + gap_y)
            canvas.paste(photo, (x, y))
            positions.append((x, y, x + pw, y + ph))

    lw = GUIDE_LINE_WIDTH
    lc = GUIDE_LINE_COLOR
    tick = 20
    for (x0, y0, x1, y1) in positions:
        for cx, cy in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
            draw.line([(cx - tick, cy), (cx + tick, cy)], fill=lc, width=lw)
            draw.line([(cx, cy - tick), (cx, cy + tick)], fill=lc, width=lw)

    return canvas


def pil_to_bytes(img: Image.Image, fmt: str = "PNG", dpi: tuple = None) -> bytes:
    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if dpi:
        save_kwargs["dpi"] = dpi
    img.save(buf, **save_kwargs)
    return buf.getvalue()


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

def init_state():
    defaults = {
        "fg_rgba": None,
        "bg_color": "#FFFFFF",
        "final_photo": None,
        "layout_img": None,
        "size_name": "标准1寸 (25×35mm)",
        "brightness": 1.0,
        "contrast": 1.0,
        "uploaded_name": None,
        "face_rect": None,       # 人脸检测结果（原图坐标）
        "offset_h": 0.0,         # 裁剪水平微调
        "offset_v": 0.0,         # 裁剪垂直微调
        "composited_full": None, # 全图合成结果（用于预览显示）
        "preview_cache_key": "", # 预览缓存键，避免重复生成
        "dpi_label": "高清 (600 DPI)",
        "smoothing": 0.0,        # 磨皮强度 0~1
        "face_brighten": 0.0,    # 面部提亮 0~1
        "saturation": 1.0,       # 饱和度 0.5~2
        "sharpness": 1.0,        # 锐度 0.5~2
        "auto_enhanced_base": None,  # 一键自动优化缓存结果
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def regenerate_photo():
    """根据当前参数重新生成证件照。"""
    fg = st.session_state.fg_rgba
    if fg is None:
        return
    # 缓存键：避免相同参数重复计算
    key = (f"{st.session_state.bg_color}|{st.session_state.size_name}|{st.session_state.dpi_label}"
           f"|{st.session_state.brightness:.2f}|{st.session_state.contrast:.2f}"
           f"|{st.session_state.smoothing:.2f}|{st.session_state.face_brighten:.2f}"
           f"|{st.session_state.saturation:.2f}|{st.session_state.sharpness:.2f}"
           f"|{st.session_state.offset_h:.2f}|{st.session_state.offset_v:.2f}")
    if key == st.session_state.preview_cache_key and st.session_state.final_photo is not None:
        return
    st.session_state.preview_cache_key = key

    # 合成全图（用于裁剪 + 预览叠框）
    if st.session_state.auto_enhanced_base is not None:
        composited = st.session_state.auto_enhanced_base
    else:
        composited = apply_background(fg, st.session_state.bg_color)

    # 人脸检测（只做一次，美颜需要人脸位置）
    if st.session_state.face_rect is None:
        st.session_state.face_rect = _detect_face_rect(composited)

    # 美颜处理（在亮度/对比度之前）
    composited = beautify_image(
        composited,
        smoothing=st.session_state.smoothing,
        face_brighten=st.session_state.face_brighten,
        saturation=st.session_state.saturation,
        sharpness=st.session_state.sharpness,
        face_rect=st.session_state.face_rect,
    )

    composited = adjust_image(composited, st.session_state.brightness, st.session_state.contrast)
    st.session_state.composited_full = composited

    target_w, target_h = get_target_size(st.session_state.size_name, st.session_state.dpi_label)
    st.session_state.final_photo = smart_crop(
        composited, target_w, target_h,
        face=st.session_state.face_rect,
        offset_h=st.session_state.offset_h,
        offset_v=st.session_state.offset_v,
    )
    st.session_state.layout_img = None


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="证件照智能生成器",
        page_icon="🪪",
        layout="centered",
    )

    init_state()

    st.title("🪪 证件照智能生成器")
    st.caption("上传照片 → AI 抠图 → 选底色/尺寸 → 微调裁剪 → 下载或排版打印")

    # ═══ 顶部：上传 + 抠图（全宽）═══
    uploaded = st.file_uploader(
        "上传照片（JPG / PNG / WEBP）",
        type=["jpg", "jpeg", "png", "webp"],
        help="正面、清晰、半身照效果最佳",
        label_visibility="collapsed",
    )

    if uploaded:
        orig_img = Image.open(uploaded).convert("RGB")
        orig_img = _limit_image_size(orig_img)  # 限制尺寸避免云端 OOM
        file_id = f"{uploaded.name}_{uploaded.size}"

        if st.session_state.uploaded_name == file_id and st.session_state.fg_rgba is not None:
            pass
        else:
            if st.button("✂️ 一键智能抠图", type="primary"):
                with st.spinner("AI 正在去除背景…"):
                    result = remove_background(orig_img)
                    if result is not None:
                        st.session_state.fg_rgba = result
                        st.session_state.final_photo = None
                        st.session_state.layout_img = None
                        st.session_state.face_rect = None
                        st.session_state.composited_full = None
                        st.session_state.auto_enhanced_base = None
                        st.session_state.offset_h = 0.0
                        st.session_state.offset_v = 0.0
                        st.session_state.preview_cache_key = ""
                        st.session_state.uploaded_name = file_id
                        st.success("抠图完成！")

    st.markdown("")

    # ═══ 主体：左栏设置 + 右栏预览 ═══
    left_col, right_col = st.columns(2, gap="large")

    # ────── 左栏：参数设置 ──────
    with left_col:
        # 底色
        st.markdown("**📌 选择底色**")
        bg_cols = st.columns(3)
        for i, (label, hex_val) in enumerate(BG_COLORS.items()):
            with bg_cols[i]:
                is_active = st.session_state.bg_color == hex_val
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"bg_{label}", type=btn_type):
                    st.session_state.bg_color = hex_val
                    st.session_state.auto_enhanced_base = None
                    st.session_state.final_photo = None
                    st.session_state.layout_img = None

        st.markdown("")

        # 尺寸 + DPI 并排
        size_dpi_l, size_dpi_r = st.columns(2)
        with size_dpi_l:
            st.markdown("**📐 尺寸**")
            size_name = st.radio(
                "选择规格",
                list(SIZES.keys()),
                index=list(SIZES.keys()).index(st.session_state.size_name),
                label_visibility="collapsed",
            )
            if size_name != st.session_state.size_name:
                st.session_state.size_name = size_name
                st.session_state.final_photo = None
                st.session_state.layout_img = None

        with size_dpi_r:
            st.markdown("**🖨️ 清晰度**")
            dpi_label = st.radio(
                "选择 DPI",
                list(DPI_OPTIONS.keys()),
                index=list(DPI_OPTIONS.keys()).index(st.session_state.dpi_label),
                label_visibility="collapsed",
            )
            if dpi_label != st.session_state.dpi_label:
                st.session_state.dpi_label = dpi_label
                st.session_state.final_photo = None
                st.session_state.layout_img = None

        target_w, target_h = get_target_size(size_name, dpi_label)
        st.caption(f"输出：{target_w} × {target_h} px")

        st.markdown("")

        # 亮度/对比度
        st.markdown("**☀️ 光线微调**")
        bc_l, bc_r = st.columns(2)
        with bc_l:
            brightness = st.slider("亮度", 0.5, 2.0, st.session_state.brightness, 0.05)
        with bc_r:
            contrast = st.slider("对比度", 0.5, 2.0, st.session_state.contrast, 0.05)
        if brightness != st.session_state.brightness or contrast != st.session_state.contrast:
            st.session_state.brightness = brightness
            st.session_state.contrast = contrast
            st.session_state.final_photo = None
            st.session_state.layout_img = None

        st.markdown("")

        # 裁剪微调
        if st.session_state.fg_rgba is not None:
            st.markdown("**✂️ 裁剪微调**")
            crop_l, crop_r = st.columns(2)
            with crop_l:
                offset_h = st.slider(
                    "水平偏移", -1.0, 1.0, st.session_state.offset_h, 0.05,
                    help="负值左移，正值右移"
                )
            with crop_r:
                offset_v = st.slider(
                    "垂直偏移", -1.0, 1.0, st.session_state.offset_v, 0.05,
                    help="负值上移，正值下移"
                )
            if offset_h != st.session_state.offset_h or offset_v != st.session_state.offset_v:
                st.session_state.offset_h = offset_h
                st.session_state.offset_v = offset_v
                st.session_state.final_photo = None
                st.session_state.layout_img = None

            if st.button("🔄 重置裁剪位置"):
                st.session_state.offset_h = 0.0
                st.session_state.offset_v = 0.0
                st.session_state.final_photo = None
                st.session_state.layout_img = None
                st.rerun()

        # 尺寸参考
        with st.expander("📋 证件照尺寸参考"):
            st.markdown("""
| 规格 | 300 DPI | 600 DPI | 实际尺寸 | 常用场景 |
|------|---------|---------|----------|----------|
| 标准1寸 | 295 × 413 | 590 × 826 | 25 × 35 mm | 简历、驾照 |
| 标准2寸 | 413 × 579 | 826 × 1158 | 35 × 49 mm | 身份证、学生证 |
| 小2寸/护照 | 390 × 567 | 780 × 1134 | 33 × 48 mm | 护照、签证 |
            """)

    # ────── 右栏：预览 + 下载 + 美颜 ──────
    with right_col:
        # 确保 regenerate_photo 被调用
        if st.session_state.fg_rgba is not None:
            try:
                regenerate_photo()
            except Exception as e:
                print(f"[regenerate] ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                st.error(f"生成预览失败: {str(e)}")
                st.code(traceback.format_exc())

        # ── 预览区 ──
        if st.session_state.final_photo is not None:
            st.markdown("**📷 证件照预览**")
            photo = st.session_state.final_photo
            tw, th = photo.size
            preview = photo.copy()
            disp_h = 360
            upscale = max(1, disp_h // th)
            if upscale > 1:
                preview = preview.resize((tw * upscale, th * upscale), Image.LANCZOS)
            st.image(preview)
            dpi_val = 600 if "600" in st.session_state.dpi_label else 300
            st.caption(f"{tw}×{th} px · {dpi_val} DPI · {size_name}")

            # 下载按钮（紧邻预览）
            dl_l, dl_r = st.columns(2)
            with dl_l:
                st.download_button(
                    label=f"⬇️ 下载证件照",
                    data=pil_to_bytes(st.session_state.final_photo, dpi=(dpi_val, dpi_val)),
                    file_name="id_photo.png",
                    mime="image/png",
                    width='stretch',
                )
            with dl_r:
                if st.button("🖨️ 生成排版图"):
                    with st.spinner("正在拼版…"):
                        st.session_state.layout_img = build_layout_image(
                            st.session_state.final_photo,
                            st.session_state.size_name,
                        )
                    st.rerun()

        elif st.session_state.fg_rgba is not None:
            st.markdown("**抠图效果预览**")
            preview = apply_background(st.session_state.fg_rgba, st.session_state.bg_color)
            preview.thumbnail((600, 900), Image.LANCZOS)
            st.image(preview)
            st.caption("调整左侧参数后预览将自动更新")

        else:
            st.image(
                "data:image/svg+xml;base64," + __import__("base64").b64encode(
                    b'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400">'
                    b'<rect width="400" height="400" fill="#f5f5f5" rx="12"/>'
                    b'<text x="200" y="200" text-anchor="middle" fill="#bbb" font-size="16" font-family="Arial">'
                    b'Upload a photo to see preview here'
                    b'</text></svg>'
                ).decode(),
                width='stretch',
            )

        # ── 排版图预览 + 下载 ──
        if st.session_state.layout_img is not None:
            st.markdown("**🖨️ 6 寸排版图**")
            lt = "3×3 九宫格" if "1寸" in size_name and "2寸" not in size_name else "2×2 四宫格"
            layout_preview = st.session_state.layout_img.copy()
            layout_preview.thumbnail((500, 750), Image.LANCZOS)
            st.image(layout_preview)
            cw, ch = st.session_state.layout_img.size
            dpi_val = 600 if "600" in st.session_state.dpi_label else 300
            st.caption(f"{lt} · {cw}×{ch}px · {dpi_val} DPI")
            st.download_button(
                label=f"⬇️ 下载排版图",
                data=pil_to_bytes(st.session_state.layout_img, dpi=(dpi_val, dpi_val)),
                file_name=f"id_photo_6inch_{dpi_val}dpi.png",
                mime="image/png",
                width='stretch',
            )

        # ── 美颜修图（折叠面板）──
        if st.session_state.fg_rgba is not None:
            with st.expander("✨ 美颜修图", expanded=False):
                PRESETS = {
                    "无美颜": {"smoothing": 0.0, "face_brighten": 0.0, "saturation": 1.0, "sharpness": 1.0},
                    "轻度美颜": {"smoothing": 0.3, "face_brighten": 0.1, "saturation": 1.05, "sharpness": 1.05},
                    "中度美颜": {"smoothing": 0.5, "face_brighten": 0.2, "saturation": 1.10, "sharpness": 1.10},
                    "深度美颜": {"smoothing": 0.8, "face_brighten": 0.3, "saturation": 1.15, "sharpness": 1.15},
                }
                preset_cols = st.columns(4)
                for i, (label, params) in enumerate(PRESETS.items()):
                    with preset_cols[i]:
                        if st.button(label, key=f"preset_{label}"):
                            for k, v in params.items():
                                st.session_state[k] = v
                            st.session_state.auto_enhanced_base = None
                            st.session_state.final_photo = None
                            st.session_state.layout_img = None
                            st.session_state.preview_cache_key = ""
                            st.rerun()

                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button("🔧 一键自动优化",
                                  help="自动白平衡 + 自适应曝光，适合光线暗淡/偏色的照片"):
                        auto_input = apply_background(st.session_state.fg_rgba, st.session_state.bg_color)
                        auto_result, suggestions = auto_enhance(auto_input)
                        st.session_state.auto_enhanced_base = auto_result
                        st.session_state.smoothing = suggestions["smoothing"]
                        st.session_state.face_brighten = suggestions["face_brighten"]
                        st.session_state.saturation = suggestions["saturation"]
                        st.session_state.sharpness = suggestions["sharpness"]
                        st.session_state.final_photo = None
                        st.session_state.layout_img = None
                        st.session_state.preview_cache_key = ""
                        st.rerun()
                with btn_cols[1]:
                    if st.button("↩️ 恢复原图",
                                  help="清除所有美颜效果，恢复到原始状态"):
                        st.session_state.auto_enhanced_base = None
                        st.session_state.smoothing = 0.0
                        st.session_state.face_brighten = 0.0
                        st.session_state.saturation = 1.0
                        st.session_state.sharpness = 1.0
                        st.session_state.brightness = 1.0
                        st.session_state.contrast = 1.0
                        st.session_state.final_photo = None
                        st.session_state.layout_img = None
                        st.session_state.preview_cache_key = ""
                        st.rerun()

                # 滑块两列布局
                s_l, s_r = st.columns(2)
                with s_l:
                    smoothing = st.slider("磨皮强度", 0.0, 1.0, st.session_state.smoothing, 0.05,
                                           help="双边滤波磨皮，仅处理皮肤区域")
                    saturation = st.slider("饱和度", 0.5, 2.0, st.session_state.saturation, 0.05)
                with s_r:
                    face_brighten = st.slider("面部提亮", 0.0, 1.0, st.session_state.face_brighten, 0.05,
                                               help="局部提亮面部，改善暗淡肤色")
                    sharpness = st.slider("锐度", 0.5, 2.0, st.session_state.sharpness, 0.05)

                changed = False
                for name, val in [("smoothing", smoothing), ("face_brighten", face_brighten),
                                  ("saturation", saturation), ("sharpness", sharpness)]:
                    if val != st.session_state[name]:
                        st.session_state[name] = val
                        changed = True
                if changed:
                    st.session_state.final_photo = None
                    st.session_state.layout_img = None

        # ── 裁剪位置预览（折叠）──
        if st.session_state.fg_rgba is not None and st.session_state.final_photo is not None:
            with st.expander("🔍 查看 AI 裁剪位置"):
                if st.session_state.composited_full is not None:
                    overlay = draw_crop_overlay(
                        st.session_state.composited_full, target_w, target_h,
                        face=st.session_state.face_rect,
                        offset_h=st.session_state.offset_h,
                        offset_v=st.session_state.offset_v,
                    )
                    overlay.thumbnail((500, 750), Image.LANCZOS)
                    st.image(overlay)
                    st.caption("绿框 = AI 裁剪区域 | 蓝框 = 检测到的人脸 | 灰区 = 被裁掉的部分")


if __name__ == "__main__":
    main()
