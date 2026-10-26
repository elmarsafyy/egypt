"""Closing slide and clip for the contest video.

The slide reuses the section-title artwork (part4/images/01.jpg) with new
text; the clip holds it for a few seconds with a soft bell and fades out, in
the same format as the part videos so it can be joined onto the full video.

    python3 make_ending.py   ->  images/end.jpg, quran_contest_ending.mp4
"""
import importlib.util
import os
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FONT = os.path.join(ROOT, "fonts", "BalooBhaijaan2.ttf")
NAVY = (20, 32, 92)
_spec = importlib.util.spec_from_file_location("part2_render", os.path.join(ROOT, "part2", "render.py"))
p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p2)
W, H, FPS, SR = p2.W, p2.H, p2.FPS, p2.SR
HOLD, FADE = 9.0, 1.2


def font(size):
    f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_name("ExtraBold")
    return f


def text(img, xy, s, size, fill, stroke=None, sw=0):
    ImageDraw.Draw(img).text(xy, s, font=font(size), fill=fill, anchor="mm", direction="rtl",
                             language="ar", stroke_width=sw, stroke_fill=stroke)


def repaint(a, region, is_text, keep, inside=None):
    """Replace old-text pixels in region with that row's background colour.
    inside(row) -> (lo, hi) limits each row to the shape's interior."""
    x0, y0, x1, y1 = region
    sub = a[y0:y1, x0:x1]
    txt = is_text(sub)
    if inside:
        for y in range(sub.shape[0]):
            lo, hi = inside(y0 + y)
            txt[y, :max(lo - x0, 0)] = False
            txt[y, max(hi - x0, 0):] = False
    m = Image.fromarray((txt * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
    txt = np.asarray(m) > 0
    full = np.zeros(a.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = txt * 255
    a[:] = cv2.inpaint(a.clip(0, 255).astype(np.uint8), full, 9, cv2.INPAINT_TELEA)
    return txt


def make_slide():
    src = Image.open(os.path.join(ROOT, "part4", "images", "01.jpg")).convert("RGB")
    a = np.asarray(src).astype(np.float32)
    yellow = lambda p: (p[..., 0] > 200) & (p[..., 1] > 150) & (p[..., 2] < 140)
    blue = lambda p: (p[..., 2] > 70) & (p[..., 2] - p[..., 0] > 40)
    # "الفقرة الرابعة": dark text on the yellow pill
    repaint(a, (392, 232, 1032, 362), lambda p: ~yellow(p), yellow)
    # "اختبر حفظك / لجزء النبأ": light text on the blue banner (stop short of
    # the presenter's raised finger on the lower right)
    orig = a.copy()

    def banner_span(y):
        xs = np.where(blue(orig[y, 150:1200]))[0]
        return (150 + xs[0] + 14, 150 + xs[-1] - 14) if len(xs) else (0, 0)

    repaint(a, (318, 428, 1068, 640), lambda p: ~blue(p), blue, banner_span)
    repaint(a, (318, 640, 985, 779), lambda p: ~blue(p), blue, banner_span)
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))

    text(img, (712, 296), "انتهت المسابقة", 96, NAVY)
    text(img, (690, 515), "دمتم في رعاية الله", 92, (255, 255, 255), (12, 24, 70), 4)
    text(img, (690, 668), "جزاكم الله خيراً", 98, (255, 214, 40), (90, 50, 0), 4)
    text(img, (680, 818), "والسلام عليكم ورحمة الله وبركاته", 44, NAVY)
    os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
    out = os.path.join(HERE, "images", "end.jpg")
    img.save(out, quality=95)
    return out


def make_clip(slide):
    base = np.asarray(p2.canvas_for(slide, False)).astype(np.float32)
    n = round((HOLD + FADE) * FPS)
    out = os.path.join(ROOT, "quran_contest_ending.mp4")
    audio = np.zeros(int(n / FPS * SR), np.float32)
    b = p2.p1.bell() * 0.8
    p = int(0.4 * SR)
    audio[p:p + len(b)] += b
    raw = os.path.join(HERE, "audio.f32")
    audio.tofile(raw)
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", raw,
         "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE)
    for f in range(n):
        t = f / FPS
        k = 1.0
        if t < 0.5:
            k = t / 0.5
        elif t > HOLD:
            k = max(0.0, 1 - (t - HOLD) / FADE)
        enc.stdin.write((base * k).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    os.remove(raw)
    print("wrote", out)


if __name__ == "__main__":
    make_clip(make_slide())
