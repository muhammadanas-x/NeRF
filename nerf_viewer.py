"""
nerf_viewer.py — Render novel views from a trained NeRF and display them live.

SETUP:
    pip install torch numpy imageio imageio-ffmpeg tqdm pillow matplotlib

    Run AFTER nerf_train.py has finished and saved nerf_weights.pth

HOW TO RUN:
    python nerf_viewer.py

WHAT IT DOES:
    1. Loads your trained weights
    2. Generates 60 camera poses orbiting the scene
    3. Renders each frame one by one
    4. Shows every frame live in a matplotlib window as it renders
    5. Saves all frames as PNGs in  renders/
    6. Stitches them into  renders/orbit.mp4  video

You can close the window at any time — rendered frames are already saved.
"""

import json, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
from tqdm import tqdm

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ SETTINGS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHTS_PATH = "nerf_weights.pth"   # must match what nerf_train.py saved
OUT_DIR      = "renders"            # where PNGs and the video are saved

N_FRAMES     = 60                   # frames in the orbit video
CAM_RADIUS   = 4.0                  # distance of camera from scene centre
CAM_ELEV_DEG = 30.0                 # how high the camera sits (degrees)
FPS          = 30                   # video frame rate

CHUNK        = 2048                 # rays per GPU batch (lower = less VRAM used)

SHOW_LIVE    = True                 # pop up a window showing each frame as it renders
                                    # set False if running on a headless server


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Copied helpers (same architecture as train)
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

def run_mlp(L, pts, dirs, pf, df):
    pe = positional_encode(pts,  pf)
    de = positional_encode(dirs, df)
    h  = pe
    for i in range(8):
        if i == 4:
            h = torch.cat([h, pe], -1)
        h = F.relu(L[f"d{i}"](h))
    sigma = F.softplus(L["sigma"](h))
    feat  = L["feat"](h)
    h2    = F.relu(L["color0"](torch.cat([feat, de], -1)))
    rgb   = torch.sigmoid(L["color1"](h2))
    return rgb, sigma

def sample_coarse(rays_o, rays_d, near, far, n):
    N  = rays_o.shape[0]
    t  = torch.linspace(near, far, n, device=rays_o.device).expand(N, n)
    pts = rays_o[:, None] + rays_d[:, None] * t[:, :, None]
    return t, pts

def sample_fine(t_c, w_c, n_fine):
    eps = 1e-5
    N   = t_c.shape[0]
    w   = w_c[:, 1:-1] + eps
    pdf = w / w.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)
    u   = torch.rand(N, n_fine, device=t_c.device)
    idx = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    lo  = (idx - 1).clamp(0);  hi = idx.clamp(max=cdf.shape[-1] - 1)
    mx  = t_c[:, 1:-1].shape[-1] - 1
    clo = torch.gather(cdf, 1, lo);          chi = torch.gather(cdf, 1, hi)
    tlo = torch.gather(t_c[:,1:-1],1,lo.clamp(max=mx))
    thi = torch.gather(t_c[:,1:-1],1,hi.clamp(max=mx))
    d   = (chi - clo).clamp(eps)
    return tlo + (u - clo) / d * (thi - tlo)

def volume_render(rgb, sigma, t, rays_d, white_bkg=True):
    s     = sigma.squeeze(-1)
    dists = torch.cat([t[:,1:]-t[:,:-1], torch.full_like(t[:,:1],1e10)], -1)
    dists = dists * rays_d.norm(-1, keepdim=True)
    alpha = 1 - torch.exp(-F.relu(s) * dists)
    T     = torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),
                                     1 - alpha + 1e-10], -1), -1)[:, :-1]
    w     = alpha * T
    color = (w[:,:,None] * rgb).sum(1)
    depth = (w * t).sum(-1)
    if white_bkg:
        color = color + (1 - w.sum(-1, keepdim=True))
    return color, depth, w


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Build rays for a camera
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_rays(H, W, focal, c2w_np):
    """
    c2w_np : (4,4) numpy camera-to-world matrix
    Returns rays_o (H*W,3) and rays_d (H*W,3) as float32 numpy.
    """
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    dirs = np.stack([(i - W*.5)/focal, -(j - H*.5)/focal, -np.ones_like(i)], -1)
    rd   = np.einsum("hwc,rc->hwr", dirs, c2w_np[:3,:3])
    rd   = rd / (np.linalg.norm(rd, -1, keepdims=True) + 1e-8)
    ro   = np.broadcast_to(c2w_np[:3,3], rd.shape).copy()
    return ro.reshape(-1,3).astype(np.float32), rd.reshape(-1,3).astype(np.float32)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Render one full image
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def render_image(Lc, Lf, c2w_np, H, W, focal, cfg, device):
    """
    Renders a single H×W image from camera pose c2w_np.
    Returns rgb (H,W,3) and depth (H,W) as numpy float32 arrays.
    """
    near, far = cfg["near"], cfg["far"]
    nc, nf    = cfg["n_coarse"], cfg["n_fine"]
    pf, df    = cfg["pos_freqs"], cfg["dir_freqs"]
    white     = cfg["white_bkg"]

    ro_np, rd_np = make_rays(H, W, focal, c2w_np)
    N = ro_np.shape[0]

    all_rgb, all_depth = [], []

    for start in range(0, N, CHUNK):
        ro = torch.from_numpy(ro_np[start:start+CHUNK]).to(device)
        rd = torch.from_numpy(rd_np[start:start+CHUNK]).to(device)
        n  = ro.shape[0]

        # coarse
        t_c, pts_c = sample_coarse(ro, rd, near, far, nc)
        pf_ = pts_c.reshape(-1,3)
        df_ = rd[:,None].expand_as(pts_c).reshape(-1,3)
        rs, ss = [], []
        for j in range(0, pf_.shape[0], CHUNK):
            r, s = run_mlp(Lc, pf_[j:j+CHUNK], df_[j:j+CHUNK], pf, df)
            rs.append(r); ss.append(s)
        rgb_c   = torch.cat(rs).reshape(n, nc, 3)
        sigma_c = torch.cat(ss).reshape(n, nc, 1)
        _, _, w_c = volume_render(rgb_c, sigma_c, t_c, rd, white)

        # fine
        t_f   = sample_fine(t_c, w_c, nf)
        t_all = torch.sort(torch.cat([t_c, t_f], -1), -1).values
        pts_f = ro[:,None] + rd[:,None] * t_all[:,:,None]
        pf_   = pts_f.reshape(-1,3)
        df_   = rd[:,None].expand_as(pts_f).reshape(-1,3)
        rs, ss = [], []
        for j in range(0, pf_.shape[0], CHUNK):
            r, s = run_mlp(Lf, pf_[j:j+CHUNK], df_[j:j+CHUNK], pf, df)
            rs.append(r); ss.append(s)
        rgb_f   = torch.cat(rs).reshape(n, nc+nf, 3)
        sigma_f = torch.cat(ss).reshape(n, nc+nf, 1)
        col, dep, _ = volume_render(rgb_f, sigma_f, t_all, rd, white)

        all_rgb.append(col.cpu().numpy())
        all_depth.append(dep.cpu().numpy())

    rgb_img   = np.clip(np.concatenate(all_rgb).reshape(H, W, 3),   0, 1)
    depth_img = np.concatenate(all_depth).reshape(H, W)
    return rgb_img, depth_img


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Generate 360° orbit camera poses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def orbit_poses(n, radius, elev_deg):
    """
    Returns a list of n (4,4) camera-to-world numpy matrices that orbit
    the origin in a full circle at a fixed elevation and radius.
    """
    poses = []
    elev  = math.radians(elev_deg)
    for k in range(n):
        ang = 2 * math.pi * k / n
        pos = np.array([radius * math.cos(elev) * math.cos(ang),
                        radius * math.cos(elev) * math.sin(ang),
                        radius * math.sin(elev)], dtype=np.float32)
        z   = pos / (np.linalg.norm(pos) + 1e-8)
        up  = np.array([0., 0., 1.])
        x   = np.cross(up, z); x_n = np.linalg.norm(x)
        x   = x / x_n if x_n > 1e-6 else np.array([1.,0.,0.])
        y   = np.cross(z, x)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3,0]=x; c2w[:3,1]=y; c2w[:3,2]=z; c2w[:3,3]=pos
        poses.append(c2w)
    return poses


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Live display using matplotlib
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_live_display():
    """Opens a matplotlib figure. Returns (fig, ax_rgb, ax_depth, im_rgb, im_depth)."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    plt.ion()
    fig = plt.figure(figsize=(11, 5), facecolor="#1a1a2e")
    fig.suptitle("NeRF Live Renderer", color="white", fontsize=14, fontweight="bold")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax, title in [(ax1, "RGB"), (ax2, "Depth")]:
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="#a0c4ff", fontsize=11)
        ax.axis("off")

    # Blank placeholders
    blank_rgb   = np.ones((100, 100, 3), dtype=np.float32)
    blank_depth = np.zeros((100, 100),   dtype=np.float32)

    im_rgb   = ax1.imshow(blank_rgb,   vmin=0, vmax=1)
    im_depth = ax2.imshow(blank_depth, cmap="plasma", vmin=0, vmax=1)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.pause(0.01)
    return fig, ax1, ax2, im_rgb, im_depth


def update_live_display(fig, ax1, ax2, im_rgb, im_depth,
                        rgb_img, depth_img, frame_idx, n_frames):
    """Refresh the matplotlib window with a new frame."""
    import matplotlib.pyplot as plt

    d = depth_img
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

    im_rgb.set_data(rgb_img)
    im_depth.set_data(d_norm)
    im_depth.set_clim(0, 1)

    ax1.set_title(f"RGB  — frame {frame_idx+1}/{n_frames}",
                  color="#a0c4ff", fontsize=11)
    ax2.set_title(f"Depth  — frame {frame_idx+1}/{n_frames}",
                  color="#a0c4ff", fontsize=11)

    fig.canvas.draw_idle()
    plt.pause(0.001)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN — load model and render everything
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def view():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load weights ──────────────────────────
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Cannot find {WEIGHTS_PATH}. Run nerf_train.py first.")
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar, np.dtype])
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]

    pdim = enc_dim(3, cfg["pos_freqs"])
    ddim = enc_dim(3, cfg["dir_freqs"])

    Lc = build_mlp(pdim, ddim, cfg["hidden_dim"]).to(device)
    Lf = build_mlp(pdim, ddim, cfg["hidden_dim"]).to(device)
    Lc.load_state_dict(ckpt["coarse"])
    Lf.load_state_dict(ckpt["fine"])
    Lc.eval(); Lf.eval()
    print(f"Loaded weights from {WEIGHTS_PATH}")

    H, W, focal = cfg["H"], cfg["W"], cfg["focal"]
    print(f"Render resolution: {W}×{H}  focal={focal:.1f}")

    # ── Output folder ─────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Camera orbit ──────────────────────────
    poses = orbit_poses(N_FRAMES, CAM_RADIUS, CAM_ELEV_DEG)

    # ── Setup live viewer ─────────────────────
    fig = ax1 = ax2 = im_rgb = im_depth = None
    if SHOW_LIVE:
        try:
            fig, ax1, ax2, im_rgb, im_depth = setup_live_display()
        except Exception as e:
            print(f"Could not open display ({e}). Saving frames only.")
            fig = None

    # ── Render loop ───────────────────────────
    frames_rgb = []
    print(f"\nRendering {N_FRAMES} frames at {W}×{H} ...")

    with torch.no_grad():
        for k, c2w in enumerate(tqdm(poses, desc="Frames")):
            rgb_img, depth_img = render_image(Lc, Lf, c2w, H, W, focal, cfg, device)

            # Save PNG
            png_path = os.path.join(OUT_DIR, f"frame_{k:03d}.png")
            iio.imwrite(png_path, (rgb_img * 255).astype(np.uint8))
            frames_rgb.append((rgb_img * 255).astype(np.uint8))

            # Save depth for first frame
            if k == 0:
                d = depth_img
                d_vis = ((d-d.min())/(d.max()-d.min()+1e-8)*255).astype(np.uint8)
                iio.imwrite(os.path.join(OUT_DIR, "depth_000.png"), d_vis)

            # Update live display
            if fig is not None:
                update_live_display(fig, ax1, ax2, im_rgb, im_depth,
                                    rgb_img, depth_img, k, N_FRAMES)

    # ── Save video ────────────────────────────
    video_path = os.path.join(OUT_DIR, "orbit.mp4")
    try:
        iio.imwrite(video_path, frames_rgb, fps=FPS, codec="libx264")
        print(f"\n✓ Video saved  → {video_path}")
    except Exception as e:
        print(f"\nCould not write video ({e}). Individual frames saved in {OUT_DIR}/")

    print(f"✓ Frames saved → {OUT_DIR}/frame_000.png … frame_{N_FRAMES-1:03d}.png")

    # Keep window open after rendering finishes
    if fig is not None:
        import matplotlib.pyplot as plt
        fig.suptitle("NeRF — Done! Close window to exit.", color="#90ee90",
                     fontsize=13, fontweight="bold")
        fig.canvas.draw_idle()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    view()