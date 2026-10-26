"""Build the question-2 slide (images/03.jpg) in the same design as question 1.

Question 2 came as two plain text screenshots (one per team, three verses
each). This starts from the question-1 artwork, keeps its frame, title and
team banners, and replaces the header, the instruction and the answer boxes.
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "fonts", "BalooBhaijaan2.ttf")
NAVY = (22, 30, 84)

LULU = ["وَالْآخِرَةُ خَيْرٌ وَأَبْقَىٰ", "فَأَنذَرْتُكُمْ نَارًا تَلَظَّىٰ", "كَلَّا إِنَّ الْإِنسَانَ لَيَطْغَىٰ"]
MARJAN = ["وَفِرْعَوْنَ ذِي الْأَوْتَادِ", "يَتِيمًا ذَا مَقْرَبَةٍ", "وَإِنَّهُ لِحُبِّ الْخَيْرِ لَشَدِيدٌ"]


def font(size, weight="ExtraBold"):
    f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_name(weight)
    return f


def text(img, xy, s, size, fill, weight="ExtraBold", max_w=None):
    if max_w:
        while size > 12 and font(size, weight).getlength(s, direction="rtl", language="ar") > max_w:
            size -= 1
    ImageDraw.Draw(img).text(xy, s, font=font(size, weight), fill=fill, anchor="mm",
                             direction="rtl", language="ar")


def erase_rows(a, x0, y0, x1, y1):
    """Fill a box by blending, row by row, the colours just left and right of it."""
    for y in range(y0, y1):
        l = a[y, x0 - 3:x0].mean(axis=0)
        r = a[y, x1:x1 + 3].mean(axis=0)
        t = np.linspace(0, 1, x1 - x0)[:, None]
        a[y, x0:x1] = l * (1 - t) + r * t


def answer_box(img, box, r=22):
    """Rounded light-blue answer box with a blue frame, like question 1's."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    S = 3
    fill = Image.new("RGB", (w, h))
    fa = np.zeros((h, w, 3), np.float32)
    t = np.linspace(0, 1, h)[:, None]
    fa[:] = (np.array([255, 255, 250]) * (1 - t) + np.array([214, 238, 252]) * t)[:, None, :]
    fill = Image.fromarray(fa.astype(np.uint8))
    m = Image.new("L", (w * S, h * S), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * S - 1, h * S - 1), r * S, fill=255)
    m = m.resize((w, h), Image.LANCZOS)
    frame = Image.new("L", (w * S, h * S), 0)
    fd = ImageDraw.Draw(frame)
    fd.rounded_rectangle((0, 0, w * S - 1, h * S - 1), r * S, fill=255)
    fd.rounded_rectangle((5 * S, 5 * S, w * S - 5 * S, h * S - 5 * S), (r - 5) * S, fill=0)
    frame = frame.resize((w, h), Image.LANCZOS)
    shadow = Image.new("L", img.size, 0)
    shadow.paste(m, (x0 + 2, y0 + 4))
    img.paste(Image.new("RGB", img.size, (150, 175, 200)), (0, 0), shadow.filter(ImageFilter.GaussianBlur(4)).point(lambda v: v // 3))
    img.paste(fill, (x0, y0), m)
    img.paste(Image.new("RGB", (w, h), (36, 150, 228)), (x0, y0), frame)
    inner = Image.new("L", (w * S, h * S), 0)
    ImageDraw.Draw(inner).rounded_rectangle((9 * S, 9 * S, w * S - 9 * S, h * S - 9 * S), (r - 8) * S, outline=255, width=2 * S)
    img.paste(Image.new("RGB", (w, h), (170, 215, 245)), (x0, y0), inner.resize((w, h), Image.LANCZOS))


def badge(img, c, n, r=34):
    """Yellow 12-point star with an Arabic-Indic numeral, like question 1's."""
    S = 4
    size = (r + 6) * 2 * S
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx = cy = size / 2
    pts = []
    for i in range(24):
        rr = (r if i % 2 == 0 else r * 0.86) * S
        ang = math.pi * i / 12
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    d.polygon(pts, fill=(236, 120, 30, 255))
    inner = [(cx + (x - cx) * 0.9, cy + (y - cy) * 0.9) for x, y in pts]
    d.polygon(inner, fill=(255, 205, 40, 255))
    d.ellipse((cx - r * 0.55 * S, cy - r * 0.7 * S, cx + r * 0.5 * S, cy), fill=(255, 238, 120, 120))
    layer = layer.resize((size // S, size // S), Image.LANCZOS)
    img.paste(layer, (c[0] - size // S // 2, c[1] - size // S // 2), layer)
    text(img, (c[0], c[1] + 2), "١٢٣"[n - 1], 46, NAVY)


def keep_foreground(img, src, box, hue_test):
    """Put back decorations (lantern, leaves) that overlapped the cleared area."""
    x0, y0, x1, y1 = box
    a = np.asarray(src.crop(box).convert("HSV")).astype(int)
    m = hue_test(a[..., 0] * 360 // 255, a[..., 1], a[..., 2])
    mask = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(5))
    img.paste(src.crop(box), (x0, y0), mask.filter(ImageFilter.GaussianBlur(1)))


def main():
    src = Image.open(os.path.join(HERE, "images", "02.jpg")).convert("RGB")
    img = src.copy()
    a = np.asarray(img).astype(np.float32)
    erase_rows(a, 562, 150, 1058, 243)    # "السؤال الأول"
    erase_rows(a, 420, 384, 1200, 454)    # instruction text
    img = Image.fromarray(a.clip(0, 255).astype(np.uint8))

    # clear the answer area and lay out three boxes per team
    ImageDraw.Draw(img).rectangle((60, 462, 1548, 770), fill=(253, 253, 245))
    top, h, gap = 468, 92, 10
    for row in range(3):
        y = top + row * (h + gap)
        answer_box(img, (822, y, 1538, y + h))       # فريق اللؤلؤ (right)
        answer_box(img, (68, y, 798, y + h))         # فريق المرجان (left)
        cy = y + h // 2
        badge(img, (1478, cy), row + 1)
        badge(img, (738, cy), row + 1)
        text(img, (1135, cy + 2), LULU[row], 40, NAVY, max_w=560)
        text(img, (385, cy + 2), MARJAN[row], 40, NAVY, max_w=580)

    # lantern (gold) on the right and leaves/books on the left sat in front
    keep_foreground(img, src, (1530, 640, 1600, 790),
                    lambda h, s, v: (h >= 20) & (h <= 60) & (s > 110))
    keep_foreground(img, src, (0, 680, 140, 790),
                    lambda h, s, v: ((h >= 70) & (h <= 160) & (s > 90)) | ((h < 40) & (s > 120) & (v < 200)))

    text(img, (805, 198), "السؤال الثاني", 80, NAVY)
    text(img, (805, 414), "أكمل آيتين بعد كل آية مما يلي", 44, NAVY)
    img.save(os.path.join(HERE, "images", "03.jpg"), quality=95)


if __name__ == "__main__":
    main()
