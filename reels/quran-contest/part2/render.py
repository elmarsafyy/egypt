"""Render part 2 (التجويد) of the "الماهرون بالقرآن" contest video.

Same look as part 1: each slide shows while its stretch of audio/voice.wav
plays, and each question gets a 5-second countdown. Question 10 has a verse
for each team, so it gets a countdown after each team's prompt.

    python3 render.py   ->  quran_contest_part2.mp4
"""
import importlib.util
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "part1_render", os.path.join(HERE, "..", "part1", "render.py"))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)
W, H, FPS, SR = p1.W, p1.H, p1.FPS, p1.SR
TIMER_SECONDS = 5
TIMER_STRIP = 300  # left strip kept clear for the countdown on question slides

# (image, [(voice start, voice end, countdown after?), ...])
SLIDES = [
    ("01.jpg", [(0.00, 1.97, False)]),        # الفقرة الثانية: التجويد
    ("02.jpg", [(2.69, 15.43, True)]),
    ("03.jpg", [(16.09, 35.36, True)]),
    ("04.jpg", [(35.98, 54.49, True)]),
    ("05.jpg", [(55.18, 75.99, True)]),
    ("06.jpg", [(76.66, 90.04, True)]),
    ("07.jpg", [(90.70, 106.25, True)]),
    ("08.jpg", [(106.75, 123.23, True)]),
    ("09.jpg", [(123.98, 141.36, True)]),
    ("10.jpg", [(142.27, 157.96, True)]),
    ("11.jpg", [(158.74, 168.47, True),       # فريق اللؤلؤ
                (169.58, 174.92, True)]),     # فريق المرجان
]
LEAD, PAD = 0.12, 0.15
PRE, POST = 0.5, 0.9
GAP_BEFORE_TIMER, AFTER_TIMER = 0.3, 1.2


def load_voice():
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", os.path.join(HERE, "audio", "voice.wav"),
         "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def canvas_for(path, question):
    """Blurred backdrop; the slide fits the frame, or the area right of the
    countdown strip on question slides, so the timer never covers it."""
    img = Image.open(path).convert("RGB")
    bg = img.resize((W, round(img.height * W / img.width)))
    if bg.height < H:
        bg = img.resize((round(img.width * H / img.height), H))
    bg = bg.crop(((bg.width - W) // 2, (bg.height - H) // 2,
                  (bg.width - W) // 2 + W, (bg.height - H) // 2 + H))
    bg = bg.filter(ImageFilter.GaussianBlur(30))
    bg = Image.blend(bg, Image.new("RGB", (W, H), (10, 20, 50)), 0.35)
    box_w = W - TIMER_STRIP if question else W
    scale = min(box_w / img.width, H / img.height)
    fg = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    x = W - fg.width if question else (W - fg.width) // 2
    bg.paste(fg, (x, (H - fg.height) // 2))
    return bg


def countdown_audio(track, at):
    for s in range(TIMER_SECONDS):
        tk = p1.tick(high=TIMER_SECONDS - s <= 2)
        p = int((at + s) * SR)
        track[p:p + len(tk)] += tk
    b = p1.bell()
    p = int((at + TIMER_SECONDS) * SR)
    track[p:p + len(b)] += b[:len(track) - p]


def main():
    voice = load_voice()
    out = os.path.join(HERE, "quran_contest_part2.mp4")
    tmp = tempfile.mkdtemp()
    audio, parts = [], []
    for idx, (name, segments) in enumerate(SLIDES):
        question = any(timer for _, _, timer in segments)
        # lay out the slide's timeline: [pre][voice][gap][countdown][after] ...
        pieces, timers, t = [], [], PRE
        for vs, ve, timer in segments:
            clip = voice[int(max(vs - LEAD, 0) * SR):int((ve + PAD) * SR)].copy()
            fade = int(0.04 * SR)
            clip[-fade:] *= np.linspace(1, 0, fade)
            pieces.append((t, clip))
            t += len(clip) / SR
            if timer:
                t += GAP_BEFORE_TIMER
                timers.append(t)
                t += TIMER_SECONDS + AFTER_TIMER
            else:
                t += POST
        n_frames = round(t * FPS)
        total = n_frames / FPS
        track = np.zeros(int(round(total * SR)), np.float32)
        for at, clip in pieces:
            p = int(at * SR)
            track[p:p + len(clip)] += clip
        for at in timers:
            countdown_audio(track, at)
        audio.append(track)

        base = canvas_for(os.path.join(HERE, "images", name), question)
        seg = os.path.join(tmp, f"{idx:02d}.mp4")
        enc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", "18", "-pix_fmt", "yuv420p", seg], stdin=subprocess.PIPE)
        base_bytes = base.tobytes()
        for f in range(n_frames):
            ft = f / FPS
            active = [at for at in timers if at - 0.25 <= ft < at + TIMER_SECONDS + AFTER_TIMER]
            if active:
                enc.stdin.write(p1.timer_frame(base, max(ft - active[0], 0)).tobytes())
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
