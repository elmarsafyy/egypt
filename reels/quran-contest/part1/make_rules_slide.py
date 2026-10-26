"""Rebuild the competition-rules slide (images/02.jpg) in the new section order.

Source: images/02_src.jpg (the original artwork). Edits, keeping its design:
  1  مفاهيم حول سورة هود          (title text only)
  2  التجويد                     (unchanged)
  3  اختبر حفظك لسورة هود         (purple panel: number, title, book label)
  4  اختبر حفظك في جزء النبأ      (green panel: rebuilt from the purple panel's
                                   stopwatch + book, recoloured green)
"""
import colorsys
import os
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = os.path.join(HERE, "..", "fonts", "BalooBhaijaan2.ttf")


def font(size, weight="ExtraBold"):
    f = ImageFont.truetype(FONT, size, layout_engine=ImageFont.Layout.RAQM)
    f.set_variation_by_name(weight)
    return f


def erase(a, x0, y0, x1, y1):
    """Fill a box by blending, row by row, the colours just left and right of it."""
    for y in range(y0, y1):
        l = a[y, x0 - 3:x0].mean(axis=0)
        r = a[y, x1:x1 + 3].mean(axis=0)
        t = np.linspace(0, 1, x1 - x0)[:, None]
        a[y, x0:x1] = l * (1 - t) + r * t


def soften(img, before, box, radius=1.5):
    """Feather the edges of an erased box so it blends with its surroundings."""
    x0, y0, x1, y1 = box
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rectangle(box, fill=255)
    m = m.filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(img, before, m)


def text(img, xy, s, size, fill, stroke=None, sw=0, anchor="mm", spacing=0):
    d = ImageDraw.Draw(img)
    f = font(size)
    d.text(xy, s, font=f, fill=fill, anchor=anchor, direction="rtl", language="ar",
           stroke_width=sw, stroke_fill=stroke, spacing=spacing)


def fit_size(s, max_w, start):
    size = start
    while size > 10:
        l, t, r, b = font(size).getbbox(s, direction="rtl", language="ar")
        if r - l <= max_w:
            return size
        size -= 1
    return size


def object_mask(crop, bg_dist=20):
    """Pixels not connected to the crop border through background-like colour."""
    a = np.asarray(crop).astype(int)
    h, w, _ = a.shape
    mx, mn = a.max(axis=2), a.min(axis=2)
    bglike = (mn > 205) & (mx - mn < 40)
    seen = np.zeros((h, w), bool)
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if bglike[y, x]:
                q.append((y, x)); seen[y, x] = True
    for y in range(h):
        for x in (0, w - 1):
            if bglike[y, x] and not seen[y, x]:
                q.append((y, x)); seen[y, x] = True
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and bglike[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    m = Image.fromarray(((~seen) * 255).astype(np.uint8))
    return m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(1.2))


def purple_to_green(crop):
    a = np.asarray(crop).astype(np.float32) / 255
    out = a.copy()
    h, w, _ = a.shape
    for y in range(h):
        for x in range(w):
            r, g, b = a[y, x]
            hh, s, v = colorsys.rgb_to_hsv(r, g, b)
            if 0.66 <= hh <= 0.92 and s > 0.12:
                hh = 0.40  # green
                out[y, x] = colorsys.hsv_to_rgb(hh, min(1, s * 1.05), v * 0.88)
    return Image.fromarray((out * 255).astype(np.uint8))


def rounded_pill(img, box, fill, border, r=26):
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = box
    d.rounded_rectangle((x0 + 3, y0 + 5, x1 + 3, y1 + 5), r, fill=(0, 0, 0, 0) if img.mode == "RGBA" else (200, 205, 190))
    d.rounded_rectangle(box, r, fill=border)
    d.rounded_rectangle((x0 + 5, y0 + 5, x1 - 5, y1 - 5), r - 4, fill=fill)
    # soft top highlight like the original pills
    hl = Image.new("L", img.size, 0)
    ImageDraw.Draw(hl).rounded_rectangle((x0 + 12, y0 + 9, x1 - 12, y0 + (y1 - y0) // 2), r - 8, fill=40)
    hl = hl.filter(ImageFilter.GaussianBlur(4))
    img.paste(Image.new("RGB", img.size, (255, 255, 255)), (0, 0), hl)


def disc(img, c, fill, r=36):
    d = ImageDraw.Draw(img)
    x, y = c
    d.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    hl = Image.new("L", img.size, 0)
    ImageDraw.Draw(hl).ellipse((x - r + 8, y - r + 4, x + r - 8, y), fill=45)
    img.paste(Image.new("RGB", img.size, (255, 255, 255)), (0, 0), hl.filter(ImageFilter.GaussianBlur(5)))


def main():
    src = Image.open(os.path.join(HERE, "images", "02_src.jpg")).convert("RGB")
    img = src.copy()
    a = np.asarray(img).astype(np.float32)

    # --- panel 4: clear and rebuild ----------------------------------------
    inner = (1090, 330, 1503, 807)
    top, bot = np.array([254, 248, 228]), np.array([250, 243, 216])
    d = ImageDraw.Draw(img)
    for y in range(inner[1], inner[3]):
        t = (y - inner[1]) / (inner[3] - inner[1])
        d.line([(inner[0], y), (inner[2], y)], fill=tuple(int(v) for v in top * (1 - t) + bot * t))
    # restore the number circle that pokes into the cleared band
    circle = src.crop((1255, 252, 1355, 335))
    cm = Image.new("L", circle.size, 0)
    ImageDraw.Draw(cm).ellipse((2, 0, 98, 96), fill=255)
    img.paste(circle, (1255, 252), cm.filter(ImageFilter.GaussianBlur(1)))
    a = np.asarray(img).astype(np.float32)

    # green two-line title pill, same size as the purple one
    rounded_pill(img, (1110, 334, 1485, 456), fill=(18, 128, 66), border=(140, 214, 120))

    # stopwatch + book from the purple panel, recoloured
    obj_box = (722, 512, 1012, 795)
    crop = src.crop(obj_box)
    mask = object_mask(crop)
    green = purple_to_green(crop)
    ox = 1296 - (obj_box[2] - obj_box[0]) // 2
    img.paste(green, (ox, obj_box[1]), mask)

    # --- fresh pills, labels and number discs over the old text ------------
    rounded_pill(img, (48, 328, 360, 408), fill=(222, 112, 4), border=(255, 214, 120))
    rounded_pill(img, (722, 334, 1046, 456), fill=(112, 40, 186), border=(205, 175, 240))
    rounded_pill(img, (802, 733, 964, 784), fill=(104, 40, 170), border=(225, 205, 245), r=14)
    disc(img, (852, 298), (118, 42, 192))
    disc(img, (1305, 300), (10, 132, 72))

    # --- new text ------------------------------------------------------------
    white, navy = (255, 255, 255), (24, 38, 96)
    s = "مفاهيم حول سورة هود"
    text(img, (204, 366), s, fit_size(s, 286, 44), white, (150, 62, 0), 2)
    text(img, (884, 366), "اختبر حفظك", 46, white, (80, 20, 140), 2)
    text(img, (884, 418), "لسورة هود", 46, white, (80, 20, 140), 2)
    text(img, (883, 757), "سورة هود", 30, white, (60, 15, 110), 1)
    text(img, (852, 300), "3", 58, white, (70, 20, 130), 2)

    text(img, (1297, 371), "اختبر حفظك", 46, white, (8, 80, 40), 2)
    text(img, (1297, 423), "في جزء النبأ", 46, white, (8, 80, 40), 2)
    text(img, (1297, 490), "وفيها 10 أسئلة", 40, navy)
    text(img, (1305, 302), "4", 58, white, (5, 80, 45), 2)

    img.save(os.path.join(HERE, "images", "02.jpg"), quality=95)


if __name__ == "__main__":
    main()
