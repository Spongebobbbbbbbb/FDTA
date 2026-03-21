import argparse
import os
import sys
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Video_depth_anything")))
from video_depth_anything.video_depth import VideoDepthAnything

DATASETS = {
    "DanceTrack": "depthmapDanceTrack",
    "SportsMOT":  "depthmapSportsMOT",
    "BFT":        "depthmapBFT",
}

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def process_sequence(model, seq_path, output_dir, input_size, device, grayscale):
    img_files = sorted(glob.glob(os.path.join(seq_path, "img1", "*.jpg")))
    if not img_files:
        return 0

    frames = []
    for p in img_files:
        img = cv2.imread(p)
        if img is None:
            print(f"  [warn] failed to read {p}, skipped")
            continue
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not frames:
        return 0

    depths, _ = model.infer_video_depth(np.array(frames), 30, input_size=input_size, device=device)

    depth_dir = os.path.join(output_dir, os.path.basename(seq_path), "depth")
    os.makedirs(depth_dir, exist_ok=True)

    global_max = np.max(depths)
    for i, depth in enumerate(depths):
        d = (depth / global_max * 255).astype(np.uint8)
        if not grayscale:
            d = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)
        cv2.imwrite(os.path.join(depth_dir, f"{i + 1:08d}.png"), d)

    return len(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  type=str, required=True)
    parser.add_argument("--dataset",    type=str, required=True, choices=DATASETS)
    parser.add_argument("--split",      type=str, default="train")
    parser.add_argument("--encoder",    type=str, default="vitl", choices=["vits", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--grayscale",  action="store_true")
    args = parser.parse_args()

    split_dir  = os.path.join(args.data_root, args.dataset, args.split)
    output_dir = split_dir
    ckpt_path  = os.path.join(os.path.dirname(__file__), "../../Video_depth_anything",
                              "checkpoints", f"video_depth_anything_{args.encoder}.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = VideoDepthAnything(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    sequences = sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)))
    total = sum(
        process_sequence(model, os.path.join(split_dir, seq), output_dir, args.input_size, device, args.grayscale)
        for seq in tqdm(sequences, desc=f"{args.dataset}/{args.split}")
    )
    print(f"Done. {len(sequences)} sequences, {total} frames -> {output_dir}")


if __name__ == "__main__":
    main()
