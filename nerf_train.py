"""
nerf_train.py — Train NeRF on the Kaggle NeRF-Blender synthetic dataset.

SETUP:
    1. Unzip the Kaggle dataset. You'll get a folder like:
         nerf_synthetic/lego/   (or just lego/ directly)
       It must contain:  transforms_train.json, train/r_0.png, train/r_1.png ...

    2. Set SCENE_DIR below to point at ONE scene folder (lego, chair, drums, etc.)

    3. pip install torch numpy imageio tqdm

    4. python nerf_train.py

    When done it saves:  nerf_weights.pth
"""

import json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ SETTINGS — only change things in this block
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCENE_DIR  = "./nerf_synthetic/nerf_synthetic/lego"   # ← path to the scene folder you unzipped
SAVE_PATH  = "nerf_weights.pth"        # where the trained weights get saved

HALF_RES   = True     # True = train on 200×200 (fast). False = 400×400 (slower, sharper)
WHITE_BKG  = True     # Blender scenes look best on a white background

NEAR       = 2.0      # closest ray distance  (2.0 works for all Blender scenes)
FAR        = 6.0      # farthest ray distance (6.0 works for all Blender scenes)
N_COARSE   = 64       # samples per ray — coarse network
N_FINE     = 128      # extra samples per ray — fine network

HIDDEN_DIM = 256      # MLP width (256 = paper quality; 128 = 3× faster, slightly blurrier)
POS_FREQS  = 10       # positional encoding bands for xyz
DIR_FREQS  = 4        # positional encoding bands for view direction

BATCH_SIZE = 1024     # rays per training step
MAX_ITERS  = 1000  # total training steps  (~1.5 hrs on RTX 4050)
LR         = 5e-4     # learning rate
LOG_EVERY  = 500      # print loss / PSNR every N steps


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 1 — Positional encoding
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def positional_encode(x, n_freqs):
    """Expand input with sin/cos at multiple frequencies so the MLP can learn fine detail."""
    parts = [x]
    for i in range(n_freqs):
        f = 2.0 ** i
        parts.append(torch.sin(f * math.pi * x))
        parts.append(torch.cos(f * math.pi * x))
    return torch.cat(parts, dim=-1)

def enc_dim(raw_dim, n_freqs):
    return raw_dim + 2 * raw_dim * n_freqs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 2 — Build the NeRF MLP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_mlp(pos_dim, dir_dim, hidden):
    """
    Returns an nn.ModuleDict of all linear layers.
    Architecture from the NeRF paper:
      • 8 layers for density (skip connection feeds position back in at layer 4)
      • 1 extra layer for colour (conditioned on view direction)
    """
    L = nn.ModuleDict()
    for i in range(8):
        in_d = pos_dim if i == 0 else (hidden + pos_dim if i == 4 else hidden)
        L[f"d{i}"] = nn.Linear(in_d, hidden)
    L["sigma"]   = nn.Linear(hidden, 1)
    L["feat"]    = nn.Linear(hidden, hidden)
    L["color0"]  = nn.Linear(hidden + dir_dim, hidden // 2)
    L["color1"]  = nn.Linear(hidden // 2, 3)
    return L


def run_mlp(L, pts, dirs, pos_freqs, dir_freqs):
    """One forward pass. pts and dirs are (N,3). Returns rgb (N,3) and sigma (N,1)."""
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 3 — Ray sampling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def sample_coarse(rays_o, rays_d, near, far, n, jitter=True):
    """Stratified sampling: divide [near,far] into n bins, pick one random point per bin."""
    N  = rays_o.shape[0]
    dev = rays_o.device
    t  = torch.linspace(near, far, n, device=dev).expand(N, n).clone()
    if jitter:
        mid   = 0.5 * (t[:, 1:] + t[:, :-1])
        upper = torch.cat([mid, t[:, -1:]], -1)
        lower = torch.cat([t[:, :1], mid],  -1)
        t     = lower + (upper - lower) * torch.rand_like(t)
    pts = rays_o[:, None] + rays_d[:, None] * t[:, :, None]
    return t, pts


def sample_fine(t_c, w_c, n_fine):
    """Draw extra samples where coarse weights are high (importance sampling)."""
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
    clo = torch.gather(cdf,        1, lo);  chi = torch.gather(cdf,        1, hi)
    tlo = torch.gather(t_c[:,1:-1],1, lo.clamp(max=mx))
    thi = torch.gather(t_c[:,1:-1],1, hi.clamp(max=mx))
    d   = (chi - clo).clamp(eps)
    return tlo + (u - clo) / d * (thi - tlo)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 4 — Volume rendering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def volume_render(rgb, sigma, t, rays_d, noise=0.0, white_bkg=True):
    """
    Classic NeRF alpha compositing.
    rgb   : (N, S, 3)   sigma : (N, S, 1)
    t     : (N, S)      rays_d: (N, 3)
    Returns pixel_rgb (N,3), depth (N,), weights (N,S)
    """
    s = sigma.squeeze(-1)
    if noise > 0:
        s = s + torch.randn_like(s) * noise
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
#  STEP 5 — Full hierarchical render pass
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_rays(Lc, Lf, rays_o, rays_d, near, far, nc, nf, pf, df,
                training=True, chunk=8192, white_bkg=True):
    """
    Lc / Lf = coarse / fine ModuleDict
    Returns dict: rgb_coarse, rgb_fine, depth_fine
    """
    N = rays_o.shape[0]

    # — coarse —
    t_c, pts_c = sample_coarse(rays_o, rays_d, near, far, nc, jitter=training)
    p_flat = pts_c.reshape(-1, 3)
    d_flat = rays_d[:, None].expand_as(pts_c).reshape(-1, 3)
    rgbs, sigs = [], []
    for i in range(0, p_flat.shape[0], chunk):
        r, s = run_mlp(Lc, p_flat[i:i+chunk], d_flat[i:i+chunk], pf, df)
        rgbs.append(r); sigs.append(s)
    rgb_c   = torch.cat(rgbs).reshape(N, nc, 3)
    sigma_c = torch.cat(sigs).reshape(N, nc, 1)
    col_c, _, w_c = volume_render(rgb_c, sigma_c, t_c, rays_d,
                                  noise=1.0 if training else 0.0,
                                  white_bkg=white_bkg)

    # — fine —
    t_f   = sample_fine(t_c, w_c.detach(), nf)
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
    col_f, dep_f, _ = volume_render(rgb_f, sigma_f, t_all, rays_d,
                                    noise=0.0, white_bkg=white_bkg)

    return {"rgb_coarse": col_c, "rgb_fine": col_f, "depth_fine": dep_f}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 6 — Load the Blender dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_rays_np(H, W, focal, c2w):
    """Build pixel rays for one camera. Returns (H*W,3), (H*W,3)."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    dirs = np.stack([(i - W*.5)/focal, -(j - H*.5)/focal, -np.ones_like(i)], -1)
    rd   = np.einsum("hwc,rc->hwr", dirs, c2w[:3,:3])
    rd   = rd / (np.linalg.norm(rd, axis=-1, keepdims=True) + 1e-8)
    ro   = np.broadcast_to(c2w[:3, 3], rd.shape).copy()
    return ro.reshape(-1,3).astype(np.float32), rd.reshape(-1,3).astype(np.float32)


def load_blender(scene_dir, split="train", half_res=True, white_bkg=True):
    """
    Loads transforms_{split}.json + images from scene_dir.
    Returns rays_o, rays_d, target_rgb  — all (N_rays, 3) float32 numpy arrays.
    Also returns  H, W, focal, poses  for later use in rendering.
    """
    meta_path = os.path.join(scene_dir, f"transforms_{split}.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Could not find {meta_path}\n"
            f"Make sure SCENE_DIR points at a folder that contains transforms_train.json"
        )

    with open(meta_path) as f:
        meta = json.load(f)

    imgs, poses = [], []
    print(f"Loading {split} images...")
    for frame in tqdm(meta["frames"]):
        rel  = frame["file_path"].lstrip("./")
        path = os.path.join(scene_dir, rel + ".png")
        img  = iio.imread(path).astype(np.float32) / 255.0   # (H,W,4) RGBA
        imgs.append(img)
        poses.append(np.array(frame["transform_matrix"], dtype=np.float32))

    imgs  = np.stack(imgs)    # (N,H,W,4)
    poses = np.stack(poses)   # (N,4,4)

    H, W  = imgs.shape[1:3]
    focal = 0.5 * W / np.tan(0.5 * float(meta["camera_angle_x"]))

    if half_res:
        from PIL import Image
        H2, W2 = H // 2, W // 2
        resized = []
        for im in imgs:
            pil = Image.fromarray((im * 255).astype(np.uint8))
            resized.append(np.array(pil.resize((W2, H2), Image.LANCZOS), dtype=np.float32) / 255.0)
        imgs  = np.stack(resized)
        H, W  = H2, W2
        focal /= 2.0

    # Alpha-composite: white or black background
    rgb = imgs[..., :3] * imgs[..., 3:] + (1.0 - imgs[..., 3:]) * (1.0 if white_bkg else 0.0)

    # Build all rays up front
    all_ro, all_rd, all_gt = [], [], []
    for i in range(len(poses)):
        ro, rd = get_rays_np(H, W, focal, poses[i])
        all_ro.append(ro)
        all_rd.append(rd)
        all_gt.append(rgb[i].reshape(-1, 3).astype(np.float32))

    rays_o     = np.concatenate(all_ro)
    rays_d     = np.concatenate(all_rd)
    target_rgb = np.concatenate(all_gt)

    print(f"  {len(poses)} images | {rays_o.shape[0]:,} rays | H={H} W={W} focal={focal:.1f}")
    return rays_o, rays_d, target_rgb, H, W, focal, poses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  STEP 7 — Training loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    rays_o, rays_d, target, H, W, focal, poses = load_blender(
        SCENE_DIR, split="train", half_res=HALF_RES, white_bkg=WHITE_BKG
    )

    # Build coarse + fine MLPs
    pdim = enc_dim(3, POS_FREQS)
    ddim = enc_dim(3, DIR_FREQS)
    Lc   = build_mlp(pdim, ddim, HIDDEN_DIM).to(device)
    Lf   = build_mlp(pdim, ddim, HIDDEN_DIM).to(device)
    params = list(Lc.parameters()) + list(Lf.parameters())
    print(f"Parameters: {sum(p.numel() for p in params):,}")

    opt   = torch.optim.Adam(params, lr=LR)
    N_all = rays_o.shape[0]
    idx   = np.arange(N_all)

    for step in range(1, MAX_ITERS + 1):
        # Exponential LR decay — halves every 250k steps
        new_lr = LR * 0.5 ** (step / 250_000) 
        for pg in opt.param_groups:
            pg["lr"] = new_lr

        # Random ray batch
        sel = np.random.choice(idx, BATCH_SIZE, replace=False)
        ro  = torch.from_numpy(rays_o[sel]).to(device)
        rd  = torch.from_numpy(rays_d[sel]).to(device)
        gt  = torch.from_numpy(target[sel]).to(device)

        out  = render_rays(Lc, Lf, ro, rd, NEAR, FAR,
                           N_COARSE, N_FINE, POS_FREQS, DIR_FREQS,
                           training=True, white_bkg=WHITE_BKG)
        loss = F.mse_loss(out["rgb_fine"], gt) + F.mse_loss(out["rgb_coarse"], gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0:
            psnr = -10 * math.log10(F.mse_loss(out["rgb_fine"], gt).item() + 1e-10)
            print(f"  [{step:6d}/{MAX_ITERS}]  loss={loss.item():.5f}  "
                  f"PSNR={psnr:.2f}dB  lr={new_lr:.2e}")

    # Save everything needed for inference
    torch.save({
        "coarse": Lc.state_dict(),
        "fine":   Lf.state_dict(),
        "config": {
            "pos_freqs": POS_FREQS, "dir_freqs": DIR_FREQS,
            "hidden_dim": HIDDEN_DIM,
            "near": NEAR, "far": FAR,
            "n_coarse": N_COARSE, "n_fine": N_FINE,
            "white_bkg": WHITE_BKG,
            "H": H, "W": W, "focal": focal,
        },
        "val_poses": poses,   # save a few poses for the viewer to use as reference
    }, SAVE_PATH)
    print(f"\n✓ Saved weights → {SAVE_PATH}")


if __name__ == "__main__":
    train()