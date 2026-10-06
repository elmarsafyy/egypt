"""Cinematic glass-shatter reel (4 s, 1080x1920, 30 fps) rendered procedurally.

Timeline
  0.00-0.25  dark glass, light sweep, slow push-in
  0.25       impact: flash, shake, cracks race outward
  0.40-0.55  pane bursts toward camera at full speed
  0.55-2.90  speed ramp into slow motion, shards drift and glint
  2.90-4.00  ramp back up, shards rush past the lens, fade out

Usage: python3 render.py [out.mp4] [--preview]
"""
import math
import os
import subprocess
import sys

import cv2
import numpy as np
from scipy import signal
from scipy.io import wavfile

W, H = 1080, 1920
FPS = 30
DUR = 4.0
NF = int(FPS * DUR)
SHUTTER = 0.5      # 180 degree shutter
SEED = 7

T0 = 0.25          # impact
T1 = 0.40          # pane releases
FOCAL = (H / 2) / math.tan(math.radians(26))
D = 4.0            # pane distance from camera
IMPACT = np.array([0.05, 0.14])
PANE = (-1.7, 1.7, -2.8, 2.8)
NEAR = 0.25
SLOW = 0.07

rng = np.random.default_rng(SEED)


def smooth(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# --------------------------------------------------------------------------- time remap
def rate(t):
    if t < T1:
        return 0.0
    if t < 0.50:
        return 1.0
    if t < 0.78:
        return 1.0 + (SLOW - 1.0) * smooth((t - 0.50) / 0.28)
    if t < 3.2:
        return SLOW
    if t < 3.5:
        return SLOW + (1.2 - SLOW) * smooth((t - 3.2) / 0.3)
    return 1.2


_tg = np.linspace(0, DUR + 0.1, 20001)
_sg = np.concatenate([[0], np.cumsum([rate(x) for x in _tg[:-1]]) * (_tg[1] - _tg[0])])


def subframes(t):
    r = max(rate(t), rate(t + 1 / FPS))
    if T0 - 0.02 <= t <= T0 + 0.12:
        return 10
    return int(np.clip(round(6 + 14 * r), 6, 22)) if r > SLOW else 6


def render_frame(t):
    n = subframes(t)
    return sum(render_sub(t + (k / n) * SHUTTER / FPS) for k in range(n)) / n


def sim_time(t):
    return float(np.interp(t, _tg, _sg))


def dust_time(t):
    if t < T0:
        return -1.0
    return min(t, T1) - T0 + sim_time(t)


def crack_front(t):
    if t < T0:
        return 0.0
    return 5.0 * (1 - math.exp(-(t - T0) / 0.05))


# --------------------------------------------------------------------------- geometry
def clip_poly(poly, x0, x1, y0, y1):
    def clip(pts, inside, inter):
        out = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            ia, ib = inside(a), inside(b)
            if ia:
                out.append(a)
                if not ib:
                    out.append(inter(a, b))
            elif ib:
                out.append(inter(a, b))
        return out

    def ix(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def iy(yc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]), yc)

    pts = [tuple(p) for p in poly]
    for inside, inter in (
        (lambda p: p[0] >= x0, ix(x0)),
        (lambda p: p[0] <= x1, ix(x1)),
        (lambda p: p[1] >= y0, iy(y0)),
        (lambda p: p[1] <= y1, iy(y1)),
    ):
        if not pts:
            return []
        pts = clip(pts, inside, inter)
    return pts


def poly_area_centroid(p):
    p = np.asarray(p)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cr = x * yn - xn * y
    a = cr.sum() / 2
    if abs(a) < 1e-9:
        return 0.0, p.mean(0)
    cx = ((x + xn) * cr).sum() / (6 * a)
    cy = ((y + yn) * cr).sum() / (6 * a)
    return abs(a), np.array([cx, cy])


def generate_shards():
    nrad = 26
    step = 2 * math.pi / nrad
    ang = np.arange(nrad) * step + rng.uniform(0, 2 * math.pi)
    ang += rng.uniform(-0.3, 0.3, nrad) * step
    radii = []
    r = 0.022
    while r < 5.0:
        radii.append(r)
        r *= 1.3
    K = len(radii)
    V = np.zeros((nrad, K, 2))
    for i in range(nrad):
        drift = 0.0
        for k in range(K):
            drift += rng.normal(0, 0.05 * step)
            a = ang[i] + drift + rng.normal(0, 0.08 * step)
            rr = radii[k] * rng.uniform(0.86, 1.14)
            V[i, k] = IMPACT + rr * np.array([math.cos(a), math.sin(a)])

    polys = []
    for i in range(nrad):
        j = (i + 1) % nrad
        polys.append([IMPACT, V[i, 0], V[j, 0]])
        k = 0
        while k < K - 1:
            stepk = 1 if k < 4 else int(rng.choice([1, 1, 2, 2, 3]))
            b = min(k + stepk, K - 1)
            side_i = [V[i, m] for m in range(k, b + 1)]
            side_j = [V[j, m] for m in range(b, k - 1, -1)]
            if b - k >= 2 and rng.random() < 0.6:
                m = int(rng.integers(k + 1, b))
                m2 = int(np.clip(m + rng.integers(-1, 2), k + 1, b - 1))
                polys.append([V[i, q] for q in range(k, m + 1)] + [V[j, q] for q in range(m2, k - 1, -1)])
                polys.append([V[i, q] for q in range(m, b + 1)] + [V[j, q] for q in range(b, m2 - 1, -1)])
            elif b - k == 1 and k > 2 and rng.random() < 0.35:
                if rng.random() < 0.5:
                    polys.append([V[i, k], V[i, b], V[j, b]])
                    polys.append([V[i, k], V[j, b], V[j, k]])
                else:
                    polys.append([V[i, k], V[i, b], V[j, k]])
                    polys.append([V[i, b], V[j, b], V[j, k]])
            else:
                polys.append(side_i + side_j)
            k = b

    shards = []
    for p in polys:
        c = clip_poly(p, *PANE)
        if len(c) < 3:
            continue
        area, cen = poly_area_centroid(c)
        if area < 1e-6:
            continue
        shards.append((np.array(c), cen, area))
    return shards


def rodrigues(axis, angle):
    """axis (N,3) unit, angle (N,) -> (N,3,3)"""
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.empty((len(angle), 3, 3))
    R[:, 0, 0] = c + x * x * C
    R[:, 0, 1] = x * y * C - z * s
    R[:, 0, 2] = x * z * C + y * s
    R[:, 1, 0] = y * x * C + z * s
    R[:, 1, 1] = c + y * y * C
    R[:, 1, 2] = y * z * C - x * s
    R[:, 2, 0] = z * x * C - y * s
    R[:, 2, 1] = z * y * C + x * s
    R[:, 2, 2] = c + z * z * C
    return R


def rand_unit(n):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# --------------------------------------------------------------------------- scene setup
shards = generate_shards()
NS = len(shards)
local = []
sid = []
for k, (poly, cen, _) in enumerate(shards):
    for q in poly:
        local.append([q[0] - cen[0], q[1] - cen[1], 0.0])
        sid.append(k)
local = np.array(local)
sid = np.array(sid)
starts = np.searchsorted(sid, np.arange(NS))
ends = np.append(starts[1:], len(sid))
cen = np.array([s[1] for s in shards])
area = np.array([s[2] for s in shards])
pos0 = np.column_stack([cen, np.full(NS, D)])
rdist = np.linalg.norm(cen - IMPACT, axis=1)
outdir = (cen - IMPACT) / np.maximum(rdist, 1e-4)[:, None]
fall = np.exp(-rdist / 1.1)
vel = np.zeros((NS, 3))
vel[:, :2] = outdir * ((0.25 + 1.4 * fall) * rng.uniform(0.6, 1.4, NS))[:, None]
vel[:, :2] += rng.normal(0, 0.15, (NS, 2))
vel[:, 2] = -(0.45 + 3.0 * fall) * rng.uniform(0.5, 1.4, NS)
omega_axis = rand_unit(NS)
omega_mag = (3.0 + 14 * fall) * rng.uniform(0.5, 1.5, NS) / np.sqrt(np.clip(area, 0.004, None) / 0.004) ** 0.25
srel = 0.035 * rdist + rng.uniform(0, 0.01, NS)
tilt_axis = rand_unit(NS)
tilt_axis[:, 2] = 0
tilt_axis /= np.linalg.norm(tilt_axis, axis=1, keepdims=True) + 1e-9
tilt_ang = rng.uniform(0.006, 0.03, NS)
GRAV = np.array([0.0, -2.5, 0.0])

ND = 3200
dust_dir = rand_unit(ND)
dust_dir[:, 2] = -np.abs(dust_dir[:, 2]) - 0.6
dust_v = dust_dir * rng.uniform(0.6, 6.0, ND)[:, None]
dust_p0 = np.column_stack([IMPACT + rng.normal(0, 0.03, (ND, 2)), np.full(ND, D)])
dust_size = rng.uniform(0.0015, 0.007, ND)
dust_phase = rng.uniform(0, 2 * math.pi, ND)
dust_spin = rng.uniform(5, 40, ND)

LIGHTS = [
    (np.array([0.6, 0.75, -0.6]), np.array([1.0, 0.68, 0.38]), 36, 1.5),   # warm key, camera side
    (np.array([-0.75, 0.35, 0.7]), np.array([0.45, 0.78, 1.0]), 22, 0.9),  # cool rim, behind
]
LIGHTS = [(l / np.linalg.norm(l), c, p, k) for l, c, p, k in LIGHTS]


# --------------------------------------------------------------------------- background
def make_background():
    s = 1.25
    bw, bh = int(W * s), int(H * s)
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    bg = np.zeros((bh, bw, 3), np.float32)
    g = yy / bh
    bg[..., 0] = 0.010 + 0.006 * g
    bg[..., 1] = 0.013 + 0.006 * g
    bg[..., 2] = 0.022 + 0.004 * g
    glow = np.exp(-(((xx - bw * 0.78) / (bw * 0.45)) ** 2 + ((yy - bh * 0.18) / (bh * 0.30)) ** 2))
    bg += glow[..., None] * np.array([0.10, 0.055, 0.025], np.float32)
    glow2 = np.exp(-(((xx - bw * 0.2) / (bw * 0.5)) ** 2 + ((yy - bh * 0.85) / (bh * 0.25)) ** 2))
    bg += glow2[..., None] * np.array([0.01, 0.035, 0.05], np.float32)
    bok = np.zeros_like(bg)
    for _ in range(38):
        x, y = rng.uniform(0, bw), rng.uniform(0, bh)
        r = rng.uniform(14, 55)
        warm = rng.random() < 0.6
        col = np.array([1.0, 0.55, 0.22]) if warm else np.array([0.25, 0.65, 0.95])
        col = col * rng.uniform(0.03, 0.16) * (0.5 + 0.8 * (1 - y / bh))
        layer = np.zeros((bh, bw), np.uint8)
        cv2.circle(layer, (int(x), int(y)), int(r), 190, -1, cv2.LINE_AA)
        cv2.circle(layer, (int(x), int(y)), int(r), 255, max(2, int(r * 0.08)), cv2.LINE_AA)
        bok += (layer.astype(np.float32) / 255)[..., None] * col.astype(np.float32)
    bok = cv2.GaussianBlur(bok, (0, 0), 4.0)
    bg += bok
    return bg


BG = make_background()
_tex = cv2.GaussianBlur(rng.random((H // 8, W // 8)).astype(np.float32), (0, 0), 3)
_tex = (_tex - _tex.min()) / (_tex.max() - _tex.min())
TEX = cv2.resize(_tex, (W, H), interpolation=cv2.INTER_CUBIC)
MAPX, MAPY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
_vr = np.sqrt(((MAPX - W / 2) / (W / 2)) ** 2 * 0.75 + ((MAPY - H / 2) / (H / 2)) ** 2)
VIGNETTE = (1 - 0.55 * smooth(_vr / 1.25) ** 1.2)[..., None].astype(np.float32)
DIAG = (MAPX * 0.55 + MAPY * 0.83)
SPECKS = np.zeros((H, W), np.float32)
for _ in range(900):
    cv2.circle(SPECKS, (int(rng.uniform(0, W)), int(rng.uniform(0, H))), int(rng.integers(1, 3)),
               float(rng.uniform(0.2, 1.0)), -1, cv2.LINE_AA)
SPECKS = cv2.GaussianBlur(SPECKS, (0, 0), 0.8)


def cam_state(t):
    if t < T1:
        cz = -0.25 + 0.25 * smooth(t / T1) if t > 0 else -0.25
        cx = 0.0
    else:
        u = (t - T1) / (DUR - T1)
        cz = 0.35 * smooth((t - T1) / 2.6) + 1.1 * smooth((t - 3.2) / 0.8)
        cx = 0.12 * smooth(u)
    cy = -0.04 * smooth((t - T1) / (DUR - T1)) if t > T1 else 0.0
    roll = 0.035 * math.sin(t * 0.9) * smooth((t - T1) / 1.0)
    shake = np.zeros(2)
    for tk, amp, dec in ((T0, 34.0, 0.18), (T1, 14.0, 0.25)):
        if t >= tk:
            e = amp * math.exp(-(t - tk) / dec)
            shake += e * np.array([math.sin(t * 97.0 + tk * 13) + 0.5 * math.sin(t * 211.0),
                                   math.cos(t * 83.0 + tk * 7) + 0.5 * math.sin(t * 173.0 + 1)])
    return np.array([cx, cy, cz]), roll, shake


def project(Pw, cam, roll, shake):
    pc = Pw - cam
    cr, sr = math.cos(roll), math.sin(roll)
    x = pc[..., 0] * cr - pc[..., 1] * sr
    y = pc[..., 0] * sr + pc[..., 1] * cr
    z = pc[..., 2]
    zs = np.maximum(z, 1e-3)
    sx = W / 2 + FOCAL * x / zs + shake[0]
    sy = H / 2 - FOCAL * y / zs + shake[1]
    return sx, sy, z


# --------------------------------------------------------------------------- render
def render_sub(t):
    cam, roll, shake = cam_state(t)
    s = sim_time(t)
    front = crack_front(t)

    # background (behind glass at infinity: tiny parallax)
    zoom = 1.0 / 1.25 * (1 + 0.04 * (cam[2] + 0.25))
    bh, bw = BG.shape[:2]
    M = np.array([[zoom, 0, W / 2 - zoom * bw / 2 + shake[0] - cam[0] * 60],
                  [0, zoom, H / 2 - zoom * bh / 2 + shake[1] + cam[1] * 60]], np.float32)
    bgf = cv2.warpAffine(BG, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    tau = np.maximum(s - srel, 0.0)
    released = (s > srel).astype(float)
    cracked = smooth((front - rdist) / 0.25)
    Rt = rodrigues(tilt_axis, tilt_ang * cracked)
    Rm = rodrigues(omega_axis, omega_mag * tau)
    R = Rm @ Rt
    pos = pos0 + vel * tau[:, None] + 0.5 * GRAV[None] * (tau ** 2)[:, None]
    Vw = np.einsum('nij,nj->ni', R[sid], local) + pos[sid]
    sx, sy, sz = project(Vw, cam, roll, shake)
    normal = -R[:, :, 2]
    pc = pos - cam
    view = -pc / np.linalg.norm(pc, axis=1, keepdims=True)
    ndv = np.abs(np.einsum('ni,ni->n', normal, view))
    fres = (1 - ndv) ** 3
    broken = smooth(tau / 0.03) * released
    spec = np.zeros((NS, 3))
    for L, col, p, k in LIGHTS:
        h = L[None] + view
        h /= np.linalg.norm(h, axis=1, keepdims=True)
        spec += (np.abs(np.einsum('ni,ni->n', normal, h)) ** p)[:, None] * col[None] * k
    spec *= broken[:, None]

    m1 = np.zeros((H, W, 4), np.float32)        # dx, dy, alpha
    sp = np.zeros((H, W, 3), np.float32)
    edge = np.zeros((H, W), np.uint8)

    zc = pc[:, 2]
    order = np.argsort(-zc)
    for k in order:
        a, b = starts[k], ends[k]
        zmin = sz[a:b].min()
        if zmin < NEAR:
            continue
        xs, ys = sx[a:b], sy[a:b]
        if xs.max() < -50 or xs.min() > W + 50 or ys.max() < -50 or ys.min() > H + 50:
            continue
        fade = smooth((zmin - NEAR) / 0.7)
        n = normal[k]
        refr = 34.0 * fade * (0.25 + 0.75 * broken[k])
        dx = -n[0] * refr
        dy = n[1] * refr
        alpha = (0.05 + 0.10 * TEX_LEVEL + broken[k] * (0.06 + 0.35 * fres[k])) * fade
        pts = np.round(np.column_stack([xs, ys]) * 16).astype(np.int32)
        cv2.fillPoly(m1, [pts], (float(dx), float(dy), float(alpha), 0.0), cv2.LINE_8, 4)
        if broken[k] > 0:
            c = spec[k] * fade * 0.55
            cv2.fillPoly(sp, [pts], (float(c[0]), float(c[1]), float(c[2])), cv2.LINE_8, 4)
            inten = (0.3 + 0.9 * fres[k] + 1.6 * float(spec[k].mean())) * fade * broken[k]
            th = int(np.clip(round(FOCAL * 0.0016 / max(zc[k], 0.3)), 1, 5))
            cv2.polylines(edge, [pts], True, int(min(255, inten * 110)), th, cv2.LINE_AA, 4)
        # cracks in the still-intact pane
        if released[k] == 0 or broken[k] < 1:
            pw = local[a:b, :2] + cen[k]
            dist = np.linalg.norm(pw - IMPACT, axis=1)
            nv = b - a
            for q in range(nv):
                q2 = (q + 1) % nv
                if max(dist[q], dist[q2]) < front:
                    x0, y0 = pw[q]
                    x1, y1 = pw[q2]
                    on_border = (abs(x0 - x1) < 1e-6 and abs(abs(x0) - PANE[1]) < 1e-6) or \
                                (abs(y0 - y1) < 1e-6 and abs(abs(y0) - PANE[3]) < 1e-6)
                    if on_border:
                        continue
                    near_imp = math.exp(-0.5 * (dist[q] + dist[q2]) / 0.6)
                    v = int(min(255, (0.55 + 1.3 * near_imp) * 110 * (1 - broken[k])))
                    cv2.line(edge, tuple(pts[q]), tuple(pts[q2]), v, 1, cv2.LINE_AA, 4)

    # refraction + glass tint
    refr = cv2.remap(bgf, MAPX + m1[..., 0], MAPY + m1[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)
    A = (m1[..., 2] * (0.75 + 0.5 * TEX))[..., None]
    tint = np.array([0.16, 0.22, 0.25], np.float32)
    img = refr * (1 - A) + tint * A
    img += sp
    img += (edge.astype(np.float32) / 110)[..., None] * np.array([0.78, 0.92, 1.0], np.float32)

    # light sweep across the intact pane
    if t < T1 + 0.15:
        c = -300 + (t / 0.45) * 2300
        dd = DIAG - c
        band = (np.exp(-(dd / 220) ** 2) * 0.10 + np.exp(-(dd / 18) ** 2) * 0.35 +
                np.exp(-((dd + 90) / 8) ** 2) * 0.18) * (1 - smooth((t - T1) / 0.15))
        band = band * (0.4 + 0.9 * TEX) + band * SPECKS * 6
        img += band[..., None] * np.array([0.9, 0.95, 1.0], np.float32)

    # glass dust
    sd = dust_time(t)
    if sd > 0:
        dp = dust_p0 + dust_v * sd + 0.5 * GRAV[None] * sd * sd
        px, py, pz = project(dp, cam, roll, shake)
        ok = (pz > 0.12) & (px > -20) & (px < W + 20) & (py > -20) & (py < H + 20)
        glint = 0.25 + 3.5 * np.abs(np.sin(dust_phase + dust_spin * sd)) ** 24
        rad = dust_size * FOCAL / np.maximum(pz, 1e-3)
        small = ok & (rad < 1.6) & (px >= 0) & (px < W - 1) & (py >= 0) & (py < H - 1)
        if small.any():
            idx = py[small].astype(np.int64) * W + px[small].astype(np.int64)
            wgt = glint[small] * np.clip(rad[small] ** 2, 0.15, 2.0)
            lay = np.bincount(idx, weights=wgt, minlength=H * W).reshape(H, W).astype(np.float32)
            img += lay[..., None] * np.array([1.0, 0.92, 0.85], np.float32)
        big = np.where(ok & (rad >= 1.6))[0]
        if len(big):
            bl = np.zeros((H, W), np.float32)
            for i in big:
                r = int(min(rad[i], 60))
                fade = 0.9 / (1 + 0.04 * r * r) * glint[i]
                cv2.circle(bl, (int(px[i]), int(py[i])), r, float(fade), -1, cv2.LINE_AA)
            bl = cv2.GaussianBlur(bl, (0, 0), 2.5)
            img += bl[..., None] * np.array([0.95, 0.9, 0.85], np.float32)
    return img


TEX_LEVEL = 0.5


def aces(x):
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def post(img, t, frame_rng):
    cam, roll, shake = cam_state(t)
    # impact flash
    if t >= T0:
        ix, iy, _ = project(np.array([IMPACT[0], IMPACT[1], D]), cam, roll, shake)
        k = math.exp(-(t - T0) / 0.035)
        d2 = (MAPX - ix) ** 2 + (MAPY - iy) ** 2
        img += (2.6 * k * np.exp(-d2 / (2 * 120 ** 2)) + 0.10 * k)[..., None] * np.array([1.0, 0.93, 0.85], np.float32)
        core = 1.2 * smooth(1 - (t - T1) / 0.2) if t < T1 + 0.2 else 0.0
        img += (core * np.exp(-d2 / (2 * 40 ** 2)))[..., None] * np.array([1.0, 1.0, 1.0], np.float32)

    # bloom
    small = cv2.resize(np.maximum(img - 0.55, 0), (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    bloom = (cv2.GaussianBlur(small, (0, 0), 3) * 0.6 + cv2.GaussianBlur(small, (0, 0), 10) * 0.5 +
             cv2.GaussianBlur(small, (0, 0), 28) * 0.45)
    img = img + cv2.resize(bloom, (W, H), interpolation=cv2.INTER_LINEAR)

    img = aces(img * 1.35)
    lum = img.mean(axis=2, keepdims=True)
    img = img + (1 - lum) ** 2 * np.array([-0.012, 0.006, 0.022], np.float32) + \
        lum ** 2 * np.array([0.035, 0.008, -0.03], np.float32)
    img = np.clip(img, 0, 1) ** (1 / 2.2)

    # chromatic aberration (spikes on impact)
    ca = 0.0025 + (0.012 * math.exp(-(t - T0) / 0.08) if t >= T0 else 0)
    out = img.copy()
    for ch, sc in ((0, 1 + ca), (2, 1 - ca)):
        Ms = np.array([[sc, 0, W / 2 * (1 - sc)], [0, sc, H / 2 * (1 - sc)]], np.float32)
        out[..., ch] = cv2.warpAffine(img[..., ch], Ms, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    img = out * VIGNETTE
    img += frame_rng.normal(0, 0.018, (H // 2, W // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
    img *= 1 - smooth((t - 3.75) / 0.25)
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- audio
def make_audio(path):
    SR = 48000
    N = int(SR * DUR)
    arng = np.random.default_rng(SEED + 1)
    dry = np.zeros((2, N))
    verb = np.zeros((2, N))

    def add(bus, sig, start, pan=0.0, gain=1.0):
        i = int(start * SR)
        if i >= N:
            return
        n = min(len(sig), N - i)
        a = (pan + 1) * math.pi / 4
        bus[0, i:i + n] += sig[:n] * gain * math.cos(a)
        bus[1, i:i + n] += sig[:n] * gain * math.sin(a)

    def band(x, lo, hi, order=4):
        sos = signal.butter(order, [lo, hi], btype='band', fs=SR, output='sos')
        return signal.sosfilt(sos, x)

    def lp(x, fc, order=4):
        return signal.sosfilt(signal.butter(order, fc, fs=SR, output='sos'), x)

    def hp(x, fc, order=4):
        return signal.sosfilt(signal.butter(order, fc, btype='high', fs=SR, output='sos'), x)

    def tvec(d):
        return np.arange(int(d * SR)) / SR

    # pre-impact suck-in whoosh
    t = tvec(T0)
    w = band(arng.normal(size=len(t)), 300, 5000) * (t / T0) ** 3
    w[-int(0.02 * SR):] *= np.linspace(1, 0, int(0.02 * SR))
    add(dry, w, 0.0, 0.0, 0.35)

    # impact: sub boom, transient click, crunch
    t = tvec(1.6)
    f = 30 + 55 * np.exp(-t / 0.07)
    boom = np.tanh(2.2 * np.sin(2 * math.pi * np.cumsum(f) / SR) * np.exp(-t / 0.5))
    add(dry, boom, T0, 0.0, 0.8)
    t = tvec(0.01)
    add(dry, arng.normal(size=len(t)) * np.exp(-t / 0.0015), T0, 0.0, 0.9)
    t = tvec(0.6)
    add(dry, band(arng.normal(size=len(t)), 700, 7000) * np.exp(-t / 0.09), T0, 0.0, 0.55)
    add(verb, band(arng.normal(size=len(t)), 300, 3000) * np.exp(-t / 0.15), T0, 0.0, 0.35)

    # shatter: glass grains
    def grain(f0, tau, amp):
        t = tvec(min(6 * tau, 0.6))
        g = np.sin(2 * math.pi * f0 * t + arng.uniform(0, 6.28)) * np.exp(-t / tau)
        g += 0.5 * np.sin(2 * math.pi * f0 * 2.76 * t) * np.exp(-t / (tau * 0.55))
        g += 0.3 * np.sin(2 * math.pi * f0 * 5.4 * t) * np.exp(-t / (tau * 0.3))
        return g * amp

    for _ in range(1100):
        on = T0 + 0.004 + arng.exponential(0.16)
        if on > 1.1:
            continue
        f0 = math.exp(arng.uniform(math.log(1800), math.log(9500)))
        g = grain(f0, arng.uniform(0.004, 0.045), arng.uniform(0.04, 0.22) * math.exp(-(on - T0) / 0.35))
        add(dry, g, on, arng.uniform(-0.8, 0.8))
        add(verb, g, on, arng.uniform(-0.8, 0.8), 0.4)
    t = tvec(1.0)
    add(dry, hp(arng.normal(size=len(t)), 3500) * np.exp(-t / 0.18), T0 + 0.01, 0.0, 0.22)
    t = tvec(0.5)
    add(dry, hp(arng.normal(size=len(t)), 1500) * np.exp(-t / 0.08), T1, 0.0, 0.25)

    # slow-motion: downward pitch sweep, rumble, slow resonant tinkles
    t = tvec(0.9)
    f = 160 * np.exp(-t / 0.35) + 38
    sw = np.sin(2 * math.pi * np.cumsum(f) / SR) * smooth(t / 0.05) * np.exp(-t / 0.6)
    add(dry, sw, 0.55, 0.0, 0.45)
    t = tvec(2.6)
    rum = lp(arng.normal(size=len(t)), 140) * 3.0
    env = smooth(t / 0.4) * (1 - 0.4 * smooth((t - 0.6) / 1.4))
    add(dry, rum * env, 0.6, 0.0, 0.5)
    t = tvec(2.4)
    air = band(arng.normal(size=len(t)), 250, 1200) * smooth(t / 0.5) * (1 - smooth((t - 1.8) / 0.6))
    add(verb, air, 0.6, 0.0, 0.08)
    for _ in range(34):
        on = arng.uniform(0.75, 3.1)
        f0 = math.exp(arng.uniform(math.log(500), math.log(2600)))
        g = grain(f0, arng.uniform(0.12, 0.35), arng.uniform(0.04, 0.11))
        add(verb, g, on, arng.uniform(-0.9, 0.9))
        add(dry, g, on, arng.uniform(-0.9, 0.9), 0.35)

    # ramp back up: reverse-style swell, then rush of shards past the lens
    t = tvec(0.45)
    sw = band(arng.normal(size=len(t)), 200, 4000) * (t / t[-1]) ** 2.5
    add(dry, sw, 3.08, 0.0, 0.45)
    t = tvec(0.5)
    rf = 26 + 40 * np.exp(-t / 0.06)
    add(dry, np.sin(2 * math.pi * np.cumsum(rf) / SR) * np.exp(-t / 0.25), 3.5, 0.0, 0.5)
    for _ in range(14):
        on = arng.uniform(3.4, 3.9)
        d = arng.uniform(0.15, 0.35)
        t = tvec(d)
        p = arng.uniform(-1, 1)
        wsh = band(arng.normal(size=len(t)), 400, 6000) * np.sin(math.pi * t / d) ** 2
        add(dry, wsh, on, p, 0.12)
    for _ in range(420):
        on = 3.5 + arng.exponential(0.15)
        if on > 3.95:
            continue
        f0 = math.exp(arng.uniform(math.log(2000), math.log(9000)))
        g = grain(f0, arng.uniform(0.004, 0.03), arng.uniform(0.03, 0.14))
        add(dry, g, on, arng.uniform(-1, 1))
        add(verb, g, on, arng.uniform(-1, 1), 0.3)

    # reverb
    ir_t = tvec(2.2)
    ir = np.stack([lp(arng.normal(size=len(ir_t)), 7000) * np.exp(-ir_t / 0.45) for _ in range(2)])
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    wet = np.stack([signal.fftconvolve(dry[c] + 2.5 * verb[c], ir[c])[:N] for c in range(2)])
    mix = dry + 0.6 * verb + 0.32 * wet
    mix = np.stack([hp(mix[c], 22, 2) for c in range(2)])
    tt = np.arange(N) / SR
    mix *= (1 - smooth((tt - 3.75) / 0.25))
    mix = np.tanh(1.4 * mix / np.abs(mix).max())
    mix = mix / np.abs(mix).max() * 0.89
    wavfile.write(path, SR, (mix.T * 32767).astype(np.int16))


# --------------------------------------------------------------------------- main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'glass_shatter_reel.mp4')
    preview = '--preview' in sys.argv
    print(f'{NS} shards, {ND} dust particles')
    frame_rng = np.random.default_rng(SEED + 2)

    if preview:
        pdir = os.path.join(os.path.dirname(out), 'preview')
        os.makedirs(pdir, exist_ok=True)
        for t in (0.1, 0.27, 0.36, 0.46, 0.6, 1.2, 2.0, 2.9, 3.4, 3.65):
            fi = int(t * FPS)
            acc = render_frame(fi / FPS)
            fr = post(acc, fi / FPS, frame_rng)
            cv2.imwrite(os.path.join(pdir, f'f_{t:.2f}.jpg'), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            print('preview', t)
        return

    wav = out.rsplit('.', 1)[0] + '.wav'
    make_audio(wav)
    ff = subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-i', wav,
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p',
        '-profile:v', 'high', '-c:a', 'aac', '-b:a', '256k', '-shortest',
        '-movflags', '+faststart', out], stdin=subprocess.PIPE)
    for fi in range(NF):
        t = fi / FPS
        acc = render_frame(t)
        ff.stdin.write(post(acc, t, frame_rng).tobytes())
        if fi % 10 == 0:
            print(f'frame {fi}/{NF}', flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print('wrote', out)


if __name__ == '__main__':
    main()
