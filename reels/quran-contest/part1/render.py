"""Render part 1 of the "الماهرون بالقرآن" contest video.

Each slide image is shown while its stretch of the recorded voice plays; every
question slide then gets a 5-second on-screen countdown with tick sounds and a
bell. Voice ranges were located by transcribing audio/voice.wav; the recording
has ~3 s of clock ticking after most questions, which the ranges leave out.

    python3 render.py   ->  quran_contest_part1.mp4
"""
import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS, SR = 1920, 1080, 30, 48000
TIMER_SECONDS = 5

# (image, voice start, voice end, is_question) — seconds into audio/voice.wav
SLIDES = [
    ("01.jpg", 0.00, 28.75, False),    # افتتاح
    ("02.jpg", None, None, False),     # نظام المسابقة — see RULES_SPLICE
    ("03.jpg", 68.77, 74.71, False),   # الفريقان
    ("04.jpg", 75.88, 78.72, False),   # الفقرة الأولى
    ("05.jpg", 79.51, 93.05, True),
    ("06.jpg", 94.18, 108.33, True),
    ("07.jpg", 114.09, 130.71, True),
    ("08.jpg", 136.33, 147.76, True),
    ("09.jpg", 153.31, 167.99, True),
    ("10.jpg", 173.44, 188.80, True),
    ("11.jpg", 194.31, 208.24, True),
    ("12.jpg", 213.76, 225.67, True),
    ("13.jpg", 231.08, 247.90, True),
    ("14.jpg", 253.05, 268.24, True),
]
# The recorded rules narration follows the old section order, so the new
# order is stitched from words in the recording; (start, end, pause after).
RULES_SPLICE = [
    (29.72, 33.80, 0.45),  # مسابقة القرآن الكريم.. بين فريقينا.. نرتقي بالقرآن
    (75.80, 78.85, 0.25),  # الفقرة الأولى: مفاهيم حول سورة هود
    (37.25, 41.60, 0.45),  # وفيها عشرة أسئلة عن سورة هود، ولكل سؤال خمس ثوان
    (42.45, 47.12, 0.30),  # الفقرة الثانية: التجويد، وفيها عشرة أسئلة في أحكام التجويد
    (39.90, 41.60, 0.45),  # ولكل سؤال خمس ثوان
    (49.92, 52.20, 0.0),   # الفقرة الثالثة: اختبر حفظك في
    (38.88, 39.56, 0.25),  # سورة هود
    (52.74, 56.60, 0.45),  # وفيها عشرة أسئلة، ولكل سؤال عشر ثوان
    (57.30, 58.62, 0.15),  # الفقرة الرابعة
    (51.10, 54.62, 0.45),  # اختبر حفظك في جزء النبأ، وفيها عشرة أسئلة
    (66.80, 68.20, 0.0),   # بالتوفيق للفريقين
]

LEAD, PAD = 0.12, 0.15          # extra voice kept before/after each range
PRE, POST = 0.5, 0.9            # silence before the voice / after a plain slide
GAP_BEFORE_TIMER, AFTER_TIMER = 0.3, 1.2

NAVY = (22, 43, 110)
GOLD = (247, 196, 38)
RED = (222, 52, 52)
FONT = "/usr/share/fonts/opentype/inter/InterDisplay-Bold.otf"


def load_voice():
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", os.path.join(HERE, "audio", "voice.wav"),
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def splice(voice, pieces):
    out = []
    edge = int(0.012 * SR)
    for s, e, pause in pieces:
        p = voice[int(s * SR):int(e * SR)].copy()
        p[:edge] *= np.linspace(0, 1, edge)
        p[-edge:] *= np.linspace(1, 0, edge)
        out += [p, np.zeros(int(pause * SR), np.float32)]
    return np.concatenate(out)


def env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / attack, 1) * np.exp(-t / decay)


def tick(high=False):
    n = int(0.12 * SR)
    t = np.arange(n) / SR
    f1, f2 = (2100, 3150) if high else (1500, 2250)
    body = 0.6 * np.sin(2 * math.pi * f1 * t) + 0.25 * np.sin(2 * math.pi * f2 * t)
    click = np.random.default_rng(1).normal(0, 1, n) * env(n, 0.0005, 0.004)
    return 0.32 * (body * env(n, 0.001, 0.018) + 0.3 * click)


def bell():
    n = int(1.6 * SR)
    t = np.arange(n) / SR
    s = sum(a * np.sin(2 * math.pi * f * t) * np.exp(-t / d)
            for f, a, d in [(880, 0.5, 0.6), (1320, 0.3, 0.45), (1760, 0.18, 0.3), (2640, 0.08, 0.2)])
    return 0.4 * s * np.minimum(t / 0.003, 1)


def canvas_for(path, question=False):
    img = Image.open(path).convert("RGB")
    bg = img.resize((W, round(img.height * W / img.width))).crop((0, 0, W, H))
    bg = bg.filter(ImageFilter.GaussianBlur(30))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (10, 20, 50)), 0.35)
    fw = round(img.width * H / img.height)
    if fw > W:
        fg = img.resize((W, round(img.height * W / img.width)), Image.LANCZOS)
    else:
        fg = img.resize((fw, H), Image.LANCZOS)
    # question slides sit flush right so the countdown gets the left strip
    x = W - fg.width if question else (W - fg.width) // 2
    bg.paste(fg, (x, (H - fg.height) // 2))
    return bg


def timer_frame(base, t):
    """Countdown overlay at time t (0..TIMER_SECONDS) seconds into the timer."""
    S = 2  # supersample for smooth edges
    r, cx, cy = 104, 150, H // 2
    size = (r + 26) * 2
    ov = Image.new("RGBA", (size * S, size * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    c = size * S // 2
    left = max(TIMER_SECONDS - t, 0)
    secs = math.ceil(left) if left > 0 else 0
    frac_in_sec = t - math.floor(t)
    pop = 1 + 0.12 * math.exp(-frac_in_sec * 10) if left > 0 else 1
    rr = r * S * pop
    accent = RED if secs <= 2 else GOLD
    d.ellipse([c - rr - 14 * S, c - rr - 14 * S, c + rr + 14 * S, c + rr + 14 * S], fill=(0, 0, 0, 70))
    d.ellipse([c - rr - 10 * S, c - rr - 10 * S, c + rr + 10 * S, c + rr + 10 * S], fill=NAVY + (255,))
    d.ellipse([c - rr + 8 * S, c - rr + 8 * S, c + rr - 8 * S, c + rr - 8 * S], fill=(255, 251, 238, 255))
    sweep = 360 * left / TIMER_SECONDS
    box = [c - rr - 3 * S, c - rr - 3 * S, c + rr + 3 * S, c + rr + 3 * S]
    d.arc(box, -90, 270, fill=(255, 255, 255, 60), width=int(16 * S))
    if sweep > 0:
        d.arc(box, -90, -90 + sweep, fill=accent + (255,), width=int(16 * S))
    font = ImageFont.truetype(FONT, int(120 * S * pop))
    label = str(secs)
    d.text((c, c + 4 * S), label, font=font, anchor="mm", fill=(accent if secs <= 2 else NAVY) + (255,))
    ov = ov.resize((size, size), Image.LANCZOS)
    frame = base.copy()
    frame.paste(ov, (cx - size // 2, cy - size // 2), ov)
    return frame


def main():
    voice = load_voice()
    out = os.path.join(HERE, "quran_contest_part1.mp4")
    audio = []
    tmp = tempfile.mkdtemp()
    parts = []
    for idx, (name, vs, ve, question) in enumerate(SLIDES):
        if vs is None:
            clip = splice(voice, RULES_SPLICE)
        else:
            clip = voice[int((vs - LEAD) * SR) if vs > LEAD else 0:int((ve + PAD) * SR)].copy()
        fade = int(0.04 * SR)
        clip[-fade:] *= np.linspace(1, 0, fade)
        pre = np.zeros(int(PRE * SR), np.float32)
        voice_len = PRE + len(clip) / SR
        if question:
            total = voice_len + GAP_BEFORE_TIMER + TIMER_SECONDS + AFTER_TIMER
        else:
            total = voice_len + POST
        n_frames = round(total * FPS)
        total = n_frames / FPS
        track = np.zeros(int(round(total * SR)), np.float32)
        track[len(pre):len(pre) + len(clip)] += clip
        timer_start = voice_len + GAP_BEFORE_TIMER
        if question:
            for s in range(TIMER_SECONDS):
                tk = tick(high=TIMER_SECONDS - s <= 2)
                p = int((timer_start + s) * SR)
                track[p:p + len(tk)] += tk
            b = bell()
            p = int((timer_start + TIMER_SECONDS) * SR)
            track[p:p + len(b)] += b[:len(track) - p]
        audio.append(track)

        base = canvas_for(os.path.join(HERE, "images", name), question)
        seg = os.path.join(tmp, f"{idx:02d}.mp4")
        enc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", seg], stdin=subprocess.PIPE)
        base_bytes = base.tobytes()
        for f in range(n_frames):
            t = f / FPS
            if question and t >= timer_start:
                enc.stdin.write(timer_frame(base, t - timer_start).tobytes())
            elif question and t >= timer_start - 0.25:
                # timer pops in just before the countdown starts
                enc.stdin.write(timer_frame(base, 0).tobytes())
            else:
                enc.stdin.write(base_bytes)
        enc.stdin.close()
        enc.wait()
        parts.append(seg)
        print(f"{name}: {total:.1f}s", flush=True)

    full = np.concatenate(audio)
    full /= max(1.0, np.abs(full).max() / 0.97)
    wav = os.path.join(tmp, "audio.f32")
    full.astype(np.float32).tofile(wav)
    listing = os.path.join(tmp, "list.txt")
    with open(listing, "w") as fh:
        fh.writelines(f"file '{p}'\n" for p in parts)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", wav,
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out],
        check=True)
    print("wrote", out, f"{len(full) / SR:.1f}s")


if __name__ == "__main__":
    main()
