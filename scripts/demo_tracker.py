#!/usr/bin/env python3
# demo_tracker_dinov3.py
# Propagate a first-frame mask using DINOv3 features + prototype matching.

import argparse
from pathlib import Path
import math
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import imageio.v3 as iio


# ----------------------------
# Utilities
# ----------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_tensor_chw_uint8(img_bgr: np.ndarray) -> torch.Tensor:
    # BGR (cv2) -> RGB, HWC uint8 -> CHW float32 (0..1)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    return t  # (C,H,W)

def normalize_imgnet(t: torch.Tensor) -> torch.Tensor:
    # t: (C,H,W) in 0..1
    mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device)[:, None, None]
    std  = torch.tensor(IMAGENET_STD,  dtype=t.dtype, device=t.device)[:, None, None]
    return (t - mean) / std

def read_video(path: str) -> list[np.ndarray]:
    # Returns list of BGR frames (cv2 style)
    frames = iio.imread(path, plugin="FFMPEG")  # (T,H,W,C), RGB uint8
    if frames is None:
        raise RuntimeError(f"Could not read video: {path}. Try: pip install imageio[ffmpeg]")
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    return frames

def auto_pad_to_multiple(img: np.ndarray, multiple: int = 16) -> tuple[np.ndarray, tuple[int,int,int,int]]:
    """Pad (H,W,...) to be divisible by 'multiple'. Returns padded image and pad (top,bottom,left,right)."""
    H, W = img.shape[:2]
    Ht = math.ceil(H / multiple) * multiple
    Wt = math.ceil(W / multiple) * multiple
    pad_b = Ht - H
    pad_r = Wt - W
    if pad_b == 0 and pad_r == 0:
        return img, (0,0,0,0)
    padded = cv2.copyMakeBorder(img, 0, pad_b, 0, pad_r, borderType=cv2.BORDER_REFLECT_101)
    return padded, (0, pad_b, 0, pad_r)

def unpad(img: np.ndarray, pad: tuple[int,int,int,int]) -> np.ndarray:
    t,b,l,r = pad
    H, W = img.shape[:2]
    return img[t: H-b if b>0 else H, l: W-r if r>0 else W]

def mask_from_similarity(sim: torch.Tensor, topk: float | int = 10, temperature: float = 0.07) -> torch.Tensor:
    """
    sim: (h,w) cosine similarity in [-1,1]
    topk: if 0<topk<=1 => fraction; if 1<topk<=100 => percentile; if >100 => absolute count of tokens
    Returns binary mask (h,w) uint8 {0,255}
    """
    s = sim.flatten()
    # Optional temperature sharpening (softmax not needed; we just scale similarity)
    s_scaled = s / max(temperature, 1e-6)

    # Decide threshold
    n = s_scaled.numel()
    if 0 < topk <= 1:  # fraction
        k = max(1, int(round(n * float(topk))))
    elif 1 < topk <= 100:  # percentage
        k = max(1, int(round(n * (float(topk)/100.0))))
    else:  # absolute count
        k = max(1, min(n, int(round(topk))))

    # Top-k selection
    thr = torch.topk(s_scaled, k).values.min()
    m = (s_scaled >= thr).reshape(sim.shape)
    return (m.byte() * 255)

def morph_radius(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0: 
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask

def overlay_contour(frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    # mask uint8 {0,255}
    color = (0, 255, 0)
    overlay = frame_bgr.copy()
    overlay[mask > 0] = (0.65*overlay[mask>0] + 0.35*np.array(color)).astype(np.uint8)
    contours, _ = cv2.findContours((mask>0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = cv2.addWeighted(overlay, alpha, frame_bgr, 1-alpha, 0)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 2)
    return out

# ----------------------------
# DINOv3 feature extraction
# ----------------------------

@torch.no_grad()
def dinov3_forward_tokens(model, img_chw: torch.Tensor) -> torch.Tensor:
    """
    img_chw: (3,H,W), float normalized
    Returns L2-normalized patch tokens as (D,h,w), torch.float32 on same device.
    Tries DINOv3 forward paths robustly.
    """
    x = img_chw.unsqueeze(0)  # (1,3,H,W)
    if hasattr(model, "forward_features"):
        out = model.forward_features(x)
        # Common DINOv3 outputs:
        # 'x_norm_patchtokens': (1, N, D) (no cls), or 'x_prenorm' / 'x_norm' / etc.
        if isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]  # (1,N,D)
            elif "x_prenorm" in out:
                tokens = out["x_prenorm"][:, 1:, :]  # drop cls
            elif "x_norm" in out:
                tokens = out["x_norm"][:, 1:, :]
            else:
                raise RuntimeError("Unexpected DINOv3 forward_features output keys.")
        else:
            # Some models may return tuple; fallback to last
            tokens = out[-1]
            if tokens.dim() == 3 and tokens.shape[1] >= 1:  # (1,N+1,D)
                tokens = tokens[:, 1:, :]
    elif hasattr(model, "get_intermediate_layers"):
        # Fall back to DINOv2-style API if present
        tokens = model.get_intermediate_layers(x, n=1, reshape=False)[0]  # (1,N+1,D)
        tokens = tokens[:, 1:, :]
    else:
        raise RuntimeError("Model does not expose forward_features or get_intermediate_layers.")

    B, N, D = tokens.shape
    # infer h,w
    # We assume square or known patch grid:
    h = w = int(math.sqrt(N))
    if h * w != N:
        # Try to infer grid from image size & patch size (16 for *16 models)
        # Default to 16; adjust if using *14 etc.
        patch = 16
        h = img_chw.shape[1] // patch
        w = img_chw.shape[2] // patch
        if h * w != N:
            raise RuntimeError(f"Cannot reshape tokens: N={N}, inferred grid {h}x{w} mismatches.")

    feats = tokens.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()  # (1,D,h,w)
    feats = F.normalize(feats, dim=1)  # L2 normalize along channels
    return feats[0]  # (D,h,w)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser("DINOv3 mask-propagation tracker")
    ap.add_argument("--video",      required=True, help="path to input video (e.g., left_seq.mp4)")
    ap.add_argument("--mask",       required=True, help="binary mask for FIRST FRAME (white=object). If not binary, >127 treated as FG.")
    ap.add_argument("--out_dir",    default="./dinov3_out", help="output directory")
    ap.add_argument("--model",      default="dinov3_vitl16",
                    help="one of: dinov3_vits16, dinov3_vits16plus, dinov3_vitb16, dinov3_vitl16, dinov3_vith16plus, dinov3_vit7b16")
    ap.add_argument("--downscale",  type=float, default=1.0, help="resize frames by this factor before feature extraction")
    ap.add_argument("--topk",       type=float, default=10.0, help="percent (0-100], fraction (0-1], or absolute token count (>100)")
    ap.add_argument("--temperature",type=float, default=0.07, help="temperature for similarity sharpening")
    ap.add_argument("--radius",     type=int,   default=0, help="morph radius (pixels) for mask cleanup")
    ap.add_argument("--save_video", action="store_true", help="save overlay video with contours")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = ensure_dir(args.out_dir)
    masks_dir = ensure_dir(out_dir / "masks")
    overlays_dir = ensure_dir(out_dir / "overlays")

    # ---- Load video ----
    frames_bgr = read_video(args.video)
    T = len(frames_bgr)
    H0, W0 = frames_bgr[0].shape[:2]

    # Optional downscale for speed/memory
    if args.downscale != 1.0:
        ds = float(args.downscale)
        frames_proc = [cv2.resize(f, (int(W0*ds), int(H0*ds)), interpolation=cv2.INTER_AREA)
                       for f in frames_bgr]
    else:
        frames_proc = frames_bgr

    H, W = frames_proc[0].shape[:2]
    print(f"[info] video loaded: T={T}, H={H}, W={W}, device={device}")

    # Ensure divisible by patch size (16 for *16 models)
    patch_size = 16
    frames_pad, pads = [], []
    for f in frames_proc:
        fp, pad = auto_pad_to_multiple(f, multiple=patch_size)
        frames_pad.append(fp)
        pads.append(pad)

    # ---- Load first-frame mask and resize/pad to match processed frame ----
    m0 = Image.open(args.mask).convert("L")
    m0 = np.array(m0)
    if args.downscale != 1.0:
        m0 = cv2.resize(m0, (W, H), interpolation=cv2.INTER_NEAREST)
    m0 = (m0 > 127).astype(np.uint8) * 255
    m0p, pad0 = auto_pad_to_multiple(m0, multiple=patch_size)  # pad mask same way
    assert pad0 == pads[0], "First-frame padding mismatch—use a mask that matches the first frame resolution."

    # ---- Load DINOv3 model ----
    # You need torch>=2, torchvision, and internet to fetch weights the first time.
    model = torch.hub.load("facebookresearch/dinov3", args.model, pretrained=True)
    model = model.to(device).eval()

    # ---- Extract prototype from frame 0 ----
    img0_chw = to_tensor_chw_uint8(frames_pad[0]).to(device)
    img0_chw = normalize_imgnet(img0_chw)
    feats0 = dinov3_forward_tokens(model, img0_chw)  # (D,h,w)
    D, h, w = feats0.shape[0], feats0.shape[1], feats0.shape[2]

    # Downsample mask to token grid
    m0_small = cv2.resize(m0p, (w, h), interpolation=cv2.INTER_NEAREST)  # (h,w) uint8
    fg = m0_small > 0
    if fg.sum() < 3:
        raise RuntimeError("Too few foreground tokens in initial mask.")

    # Build prototype vector (mean of foreground token features)
    F0 = feats0.permute(1, 2, 0).reshape(-1, D)  # (h*w, D)
    proto = F0[fg.reshape(-1)].mean(dim=0, keepdim=True)         # (1,D)
    proto = F.normalize(proto, dim=1)                            # (1,D)

    # ---- Track through all frames ----
    masks_full_res = []
    overlays_full_res = []

    # Prepare writer if needed
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_dir / "dinov3_overlay.mp4"), fourcc, 30, (W0, H0))

    for t in range(T):
        # Features
        img_chw = to_tensor_chw_uint8(frames_pad[t]).to(device)
        img_chw = normalize_imgnet(img_chw)
        feats = dinov3_forward_tokens(model, img_chw)  # (D,h,w)
        feats = feats.permute(1, 2, 0).reshape(-1, D)  # (h*w, D)

        # Cosine similarity vs prototype
        sim = (feats @ proto.t()).reshape(h, w).squeeze(0)  # (h,w)

        # Build mask on token grid
        mask_small = mask_from_similarity(sim, topk=args.topk, temperature=args.temperature).cpu().numpy()

        # Upsample to padded frame size, then unpad to processed size
        mask_pad = cv2.resize(mask_small, (frames_pad[t].shape[1], frames_pad[t].shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_pad = morph_radius(mask_pad, args.radius)
        mask_proc = unpad(mask_pad, pads[t])  # (H,W) uint8

        # Upsample to ORIGINAL resolution for saving/comparison
        if (H, W) != (H0, W0):
            mask_orig = cv2.resize(mask_proc, (W0, H0), interpolation=cv2.INTER_NEAREST)
        else:
            mask_orig = mask_proc

        masks_full_res.append(mask_orig)

        # Overlay on original frame
        overlay = overlay_contour(frames_bgr[t], mask_orig, alpha=0.6)
        overlays_full_res.append(overlay)

        # Save per-frame mask & overlay
        Image.fromarray(mask_orig).save(masks_dir / f"mask_{t:04d}.png")
        cv2.imwrite(str(overlays_dir / f"overlay_{t:04d}.jpg"), overlay)

        if args.save_video:
            writer.write(overlay)

    if args.save_video:
        writer.release()

    print(f"[done] Saved {len(masks_full_res)} masks to {masks_dir}")
    if args.save_video:
        print(f"[done] Saved overlay video to {out_dir/'dinov3_overlay.mp4'}")
    print(f"[done] Per-frame overlays saved to {overlays_dir}")


if __name__ == "__main__":
    main()
