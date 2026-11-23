"""
nerf_infer.py — Render novel views from a trained NeRF checkpoint.

SETUP:
    Same environment as training:
        pip install torch numpy imageio tqdm pillow

USAGE:
    # Render a 360-degree orbit video (default):
    python nerf_infer.py

    # Render a single view at a specific elevation and radius:
    python nerf_infer.py --mode single --elevation 20 --radius 4.0

    # Render depth maps alongside RGB:
    python nerf_infer.py --depth

    # Override output path:
    python nerf_infer.py --out my_render.mp4

OUTPUT:
    nerf_orbit.mp4     — 360° turntable video  (default)
    nerf_single.png    — single-frame render    (--mode single)
    nerf_depth.mp4     — depth video            (--mode orbit --depth)
"""

import argparse, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHTS_PATH = "nerf_weights.pth"   # checkpoint saved by nerf_train.py
RENDER_CHUNK = 4096                 # rays per forward pass (lower = less VRAM)
N_ORBIT      = 40                   # frames in the orbit video
FPS          = 24                   # video frame rate
ELEVATION    = 30.0                 # default camera elevation in degrees
RADIUS       = 4.0                  # default camera distance from origin


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Shared helpers  (must match nerf_train.py exactly)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def positional_encode(x, n_freqs):
    parts = [x]
    for i in range(n_freqs):
        f = 2.0 ** i
        parts.append(torch.sin(f * math.pi * x))
        parts.append(torch.cos(f * math.pi * x))
    return torch.cat(parts, dim=-1)

def enc_dim(raw_dim, n_freqs):
    return raw_dim + 2 * raw_dim * n_freqs

def build_mlp(pos_dim, dir_dim, hidden):
    L = nn.ModuleDict()
    for i in range(8):
        in_d = pos_dim if i == 0 else (hidden + pos_dim if i == 4 else hidden)
        L[f"d{i}"] = nn.Linear(in_d, hidden)
    L["sigma"]  = nn.Linear(hidden, 1)
    L["feat"]   = nn.Linear(hidden, hidden)
    L["color0"] = nn.Linear(hidden + dir_dim, hidden // 2)
    L["color1"] = nn.Linear(hidden // 2, 3)
    return L

def run_mlp(L, pts, dirs, pos_freqs, dir_freqs):
    pe = positional_encode(pts,  pos_freqs)
    de = positional_encode(dirs, dir_freqs)
    h = pe
    for i in range(8):
        if i == 4:
            h = torch.cat([h, pe], dim=-1)
        h = F.relu(L[f"d{i}"](h))
    sigma = F.softplus(L["sigma"](h))
    feat  = L["feat"](h)
    h2    = F.relu(L["color0"](torch.cat([feat, de], dim=-1)))
    rgb   = torch.sigmoid(L["color1"](h2))
    return rgb, sigma

def sample_coarse(rays_o, rays_d, near, far, n):
    N  = rays_o.shape[0]
    dev = rays_o.device
    t   = torch.linspace(near, far, n, device=dev).expand(N, n)
    pts = rays_o[:, None] + rays_d[:, None] * t[:, :, None]
    return t, pts

def sample_fine(t_c, w_c, n_fine):
    eps = 1e-5
    N   = t_c.shape[0]
    dev = t_c.device
    w   = w_c[:, 1:-1] + eps
    pdf = w / w.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
    u   = torch.rand(N, n_fine, device=dev)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    lo  = (idx - 1).clamp(0)
    hi  = idx.clamp(max=cdf.shape[-1] - 1)
    mx  = t_c[:, 1:-1].shape[-1] - 1
    clo = torch.gather(cdf,         1, lo);  chi = torch.gather(cdf,         1, hi)
    tlo = torch.gather(t_c[:, 1:-1], 1, lo.clamp(max=mx))
    thi = torch.gather(t_c[:, 1:-1], 1, hi.clamp(max=mx))
    d   = (chi - clo).clamp(eps)
    return tlo + (u - clo) / d * (thi - tlo)

def volume_render(rgb, sigma, t, rays_d, white_bkg=True):
    s     = sigma.squeeze(-1)
    dists = torch.cat([t[:, 1:] - t[:, :-1],
                       torch.full_like(t[:, :1], 1e10)], -1)
    dists = dists * rays_d.norm(dim=-1, keepdim=True)
    alpha = 1 - torch.exp(-F.relu(s) * dists)
    T     = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]),
                                     1 - alpha + 1e-10], -1), -1)[:, :-1]
    w     = alpha * T
    color = (w[:, :, None] * rgb).sum(1)
    depth = (w * t).sum(-1)
    if white_bkg:
        color = color + (1 - w.sum(-1, keepdim=True))
    return color, depth, w


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Inference-only render  (no noise, no jitter)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@torch.no_grad()
def render_rays_infer(Lc, Lf, rays_o, rays_d, cfg):
    """
    Render a batch of rays at inference time.
    Returns rgb (N,3) and depth (N,) tensors.
    """
    near      = cfg["near"];      far    = cfg["far"]
    nc        = cfg["n_coarse"];  nf     = cfg["n_fine"]
    pf        = cfg["pos_freqs"]; df     = cfg["dir_freqs"]
    white_bkg = cfg["white_bkg"]
    chunk     = RENDER_CHUNK
    N         = rays_o.shape[0]

    # — coarse pass —
    t_c, pts_c = sample_coarse(rays_o, rays_d, near, far, nc)
    p_flat = pts_c.reshape(-1, 3)
    d_flat = rays_d[:, None].expand_as(pts_c).reshape(-1, 3)
    rgbs, sigs = [], []
    for i in range(0, p_flat.shape[0], chunk):
        r, s = run_mlp(Lc, p_flat[i:i+chunk], d_flat[i:i+chunk], pf, df)
        rgbs.append(r); sigs.append(s)
    rgb_c   = torch.cat(rgbs).reshape(N, nc, 3)
    sigma_c = torch.cat(sigs).reshape(N, nc, 1)
    _, _, w_c = volume_render(rgb_c, sigma_c, t_c, rays_d, white_bkg=white_bkg)

    # — fine pass —
    t_f   = sample_fine(t_c, w_c, nf)
    t_all = torch.sort(torch.cat([t_c, t_f], -1), -1).values
    pts_f = rays_o[:, None] + rays_d[:, None] * t_all[:, :, None]
    p_flat = pts_f.reshape(-1, 3)
    d_flat = rays_d[:, None].expand_as(pts_f).reshape(-1, 3)
    rgbs, sigs = [], []
    for i in range(0, p_flat.shape[0], chunk):
        r, s = run_mlp(Lf, p_flat[i:i+chunk], d_flat[i:i+chunk], pf, df)
        rgbs.append(r); sigs.append(s)
    rgb_f   = torch.cat(rgbs).reshape(N, nc + nf, 3)
    sigma_f = torch.cat(sigs).reshape(N, nc + nf, 1)
    color, depth, _ = volume_render(rgb_f, sigma_f, t_all, rays_d, white_bkg=white_bkg)
    return color, depth


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Render a full image from a camera-to-world pose
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_rays(H, W, focal, c2w, device):
    """Build pixel rays for one camera pose. Returns (H*W,3), (H*W,3) tensors."""
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], dim=-1)          # (H,W,3)
    c2w_t = torch.tensor(c2w, dtype=torch.float32, device=device) if isinstance(c2w, np.ndarray) else c2w
    rd    = (dirs[..., None, :] * c2w_t[:3, :3]).sum(-1)       # (H,W,3)
    rd    = rd / rd.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ro    = c2w_t[:3, 3].expand_as(rd)
    return ro.reshape(-1, 3), rd.reshape(-1, 3)


@torch.no_grad()
def render_image(Lc, Lf, c2w, cfg, device, ray_chunk=RENDER_CHUNK):
    """
    Render a full (H×W) image from pose c2w.
    Returns rgb (H,W,3) uint8 and depth (H,W) float32.
    """
    H, W, focal = cfg["H"], cfg["W"], cfg["focal"]
    ro, rd = get_rays(H, W, focal, c2w, device)
    N      = ro.shape[0]

    all_rgb, all_dep = [], []
    for i in tqdm(range(0, N, ray_chunk), desc="  rays", leave=False):
        rgb, dep = render_rays_infer(Lc, Lf, ro[i:i+ray_chunk], rd[i:i+ray_chunk], cfg)
        all_rgb.append(rgb.cpu())
        all_dep.append(dep.cpu())

    rgb_img = torch.cat(all_rgb).reshape(H, W, 3).numpy()
    dep_img = torch.cat(all_dep).reshape(H, W).numpy()
    rgb_u8  = (np.clip(rgb_img, 0, 1) * 255).astype(np.uint8)
    return rgb_u8, dep_img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Camera pose helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def pose_spherical(theta_deg, phi_deg, radius):
    """
    Build a camera-to-world matrix for a camera on a sphere.
    theta : azimuth  (0–360°)
    phi   : elevation (degrees above the equator; positive = up)
    """
    theta = math.radians(theta_deg)
    phi   = math.radians(phi_deg)

    # Camera position in world space
    x = radius * math.cos(phi) * math.sin(theta)
    y = radius * math.sin(phi)
    z = radius * math.cos(phi) * math.cos(theta)
    pos = np.array([x, y, z], dtype=np.float32)

    # Look-at construction
    forward = -pos / (np.linalg.norm(pos) + 1e-8)   # points toward origin
    world_up = np.array([0., 1., 0.], dtype=np.float32)
    right    = np.cross(world_up, -forward)
    right   /= np.linalg.norm(right) + 1e-8
    up       = np.cross(-forward, right)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward   # OpenGL convention: camera looks down -Z
    c2w[:3, 3] = pos
    return c2w


def depth_to_uint8(dep):
    """Normalise a depth map to a grey-scale uint8 image."""
    lo, hi = dep.min(), dep.max()
    if hi - lo < 1e-6:
        return np.zeros_like(dep, dtype=np.uint8)
    norm = (dep - lo) / (hi - lo)
    grey = (norm * 255).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)   # (H,W,3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Load checkpoint
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_checkpoint(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Train first with:  python nerf_train.py"
        )
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    pdim = enc_dim(3, cfg["pos_freqs"])
    ddim = enc_dim(3, cfg["dir_freqs"])
    Lc   = build_mlp(pdim, ddim, cfg["hidden_dim"]).to(device)
    Lf   = build_mlp(pdim, ddim, cfg["hidden_dim"]).to(device)
    Lc.load_state_dict(ckpt["coarse"])
    Lf.load_state_dict(ckpt["fine"])
    Lc.eval(); Lf.eval()

    val_poses = ckpt.get("val_poses", None)
    print(f"✓ Loaded checkpoint: {path}")
    print(f"  H={cfg['H']} W={cfg['W']} focal={cfg['focal']:.1f}  "
          f"near={cfg['near']} far={cfg['far']}")
    return Lc, Lf, cfg, val_poses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Render modes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_orbit(Lc, Lf, cfg, device, args):
    """Render a 360° orbit video and save to MP4."""
    out_rgb   = args.out or "nerf_orbit.mp4"
    out_depth = out_rgb.replace(".mp4", "_depth.mp4")

    azimuths = np.linspace(0, 360, N_ORBIT, endpoint=False)
    frames_rgb, frames_dep = [], []

    print(f"Rendering {N_ORBIT} frames  (elevation={args.elevation}°, radius={args.radius})")
    for az in tqdm(azimuths, desc="frames"):
        c2w = pose_spherical(az, args.elevation, args.radius)
        rgb, dep = render_image(Lc, Lf, c2w, cfg, device)
        frames_rgb.append(rgb)
        if args.depth:
            frames_dep.append(depth_to_uint8(dep))

    # Write RGB video
    iio.imwrite(out_rgb, np.stack(frames_rgb), fps=FPS, codec="libx264", plugin="pyav")
    print(f"✓ Saved orbit video → {out_rgb}")

    if args.depth:
        iio.imwrite(out_depth, np.stack(frames_dep), fps=FPS, codec="libx264",
                    quality=8, plugin="pyav")
        print(f"✓ Saved depth video → {out_depth}")


def render_single(Lc, Lf, cfg, device, args):
    """Render one image and save as PNG."""
    out = args.out or "nerf_single.png"
    print(f"Rendering single frame  (elevation={args.elevation}°, radius={args.radius})")
    c2w = pose_spherical(args.azimuth, args.elevation, args.radius)
    rgb, dep = render_image(Lc, Lf, c2w, cfg, device)
    iio.imwrite(out, rgb)
    print(f"✓ Saved image → {out}")

    if args.depth:
        dep_path = out.replace(".png", "_depth.png")
        iio.imwrite(dep_path, depth_to_uint8(dep))
        print(f"✓ Saved depth map → {dep_path}")


def render_training_views(Lc, Lf, cfg, val_poses, device, args):
    """
    Re-render a subset of the training poses (useful to check quality).
    Saves individual PNGs into ./rendered_views/.
    """
    out_dir = args.out or "rendered_views"
    os.makedirs(out_dir, exist_ok=True)
    n = min(len(val_poses), args.n_views)
    print(f"Re-rendering {n} training view(s) → {out_dir}/")
    for i in tqdm(range(n), desc="views"):
        rgb, dep = render_image(Lc, Lf, val_poses[i], cfg, device)
        iio.imwrite(os.path.join(out_dir, f"view_{i:03d}.png"), rgb)
        if args.depth:
            iio.imwrite(os.path.join(out_dir, f"view_{i:03d}_depth.png"),
                        depth_to_uint8(dep))
    print(f"✓ Done — {n} images saved in {out_dir}/")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Entry point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    p = argparse.ArgumentParser(description="NeRF inference — render novel views.")
    p.add_argument("--weights",   default=WEIGHTS_PATH,
                   help="Path to checkpoint (default: nerf_weights.pth)")
    p.add_argument("--mode",      default="orbit",
                   choices=["orbit", "single", "training"],
                   help="orbit: 360° video | single: one PNG | training: re-render training poses")
    p.add_argument("--out",       default=None,
                   help="Output file / directory (auto-named if omitted)")
    p.add_argument("--elevation", type=float, default=ELEVATION,
                   help="Camera elevation in degrees (default: 30)")
    p.add_argument("--azimuth",   type=float, default=0.0,
                   help="[single mode] Azimuth in degrees (default: 0)")
    p.add_argument("--radius",    type=float, default=RADIUS,
                   help="Camera distance from origin (default: 4.0)")
    p.add_argument("--depth",     action="store_true",
                   help="Also save depth map(s)")
    p.add_argument("--n_views",   type=int, default=10,
                   help="[training mode] How many training views to re-render (default: 10)")
    p.add_argument("--chunk",     type=int, default=RENDER_CHUNK,
                   help="Rays per GPU batch (lower = less VRAM, default: 4096)")
    return p.parse_args()


def main():
    args   = parse_args()
    global RENDER_CHUNK
    RENDER_CHUNK = args.chunk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    Lc, Lf, cfg, val_poses = load_checkpoint(args.weights, device)

    if args.mode == "orbit":
        render_orbit(Lc, Lf, cfg, device, args)
    elif args.mode == "single":
        render_single(Lc, Lf, cfg, device, args)
    elif args.mode == "training":
        if val_poses is None:
            raise RuntimeError("Checkpoint has no saved poses. Re-train to include them.")
        render_training_views(Lc, Lf, cfg, val_poses, device, args)


if __name__ == "__main__":
    main()