#!/usr/bin/env python3
"""
tools/capture_shape_masks.py

Real-segmented-mask collection tool for building an actual (not
synthetic) shape-CNN test set -- the single most valuable diagnostic
missing from the current benchmark. Every number `tools/benchmark_shape_cnn.py`
produces is on SYNTHETIC data; this tool lets you build the real-camera
test set the plan calls for, using your own webcam and live_demo.py's
existing colour/motion segmentation pipeline.

IMPORTANT: this tool has NOT been tested against a live camera in this
environment (no camera access here) -- only its file-saving logic and
import wiring have been verified against synthetic frames. Please treat
the first few captures as a smoke test of the tool itself before relying
on it for a real data-collection session.

Usage
-----
    python3 tools/capture_shape_masks.py --source camera --out dataset/real

Controls (in addition to everything live_demo.py already supports --
colour/motion mode, presets, 's' to sample colour, etc.):
    1     save the current selected mask as a Circle example
    2     save the current selected mask as a Square example
    3     save the current selected mask as a Rectangle example
    0     save as "Other" (misc/unknown -- for future use, not yet
          part of the 3-class model, but worth collecting anyway)
    q     quit

Each press saves:
    <out>/<class>/<NNNN>.npy   -- the raw boolean self.selected_mask
                                   (full-frame-sized, NOT cropped -- crop
                                   at load time to keep the raw capture
                                   maximally reusable)
    <out>/<class>/<NNNN>.png   -- a quick human-viewable render of the
                                   saved mask, for a sanity-check pass
                                   before using the data (delete any
                                   capture that clearly segmented the
                                   wrong thing)

A running per-class capture count is shown in the HUD so you know when
you've hit the plan's suggested 100-300-per-class target.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location("live_demo", str(PROJECT_ROOT / "live_demo.py"))
_live_demo = importlib.util.module_from_spec(_spec)
sys.modules["live_demo"] = _live_demo
_spec.loader.exec_module(_live_demo)

VisionPipeline = _live_demo.VisionPipeline
make_source = _live_demo.make_source
PYGAME_AVAILABLE = _live_demo.PYGAME_AVAILABLE
PYGAME_ERROR = _live_demo.PYGAME_ERROR

CLASS_KEYS = {
    "1": "circle",
    "2": "square",
    "3": "rectangle",
    "0": "other",
}


def next_index(class_dir: Path) -> int:
    existing = list(class_dir.glob("*.npy"))
    if not existing:
        return 1
    nums = [int(p.stem) for p in existing if p.stem.isdigit()]
    return (max(nums) + 1) if nums else 1


def save_mask(mask: np.ndarray, out_root: Path, class_name: str) -> tuple:
    class_dir = out_root / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(class_dir)
    npy_path = class_dir / f"{idx:04d}.npy"
    np.save(npy_path, mask.astype(bool))

    try:
        png_path = class_dir / f"{idx:04d}.png"
        import pygame
        surf = pygame.Surface((mask.shape[1], mask.shape[0]))
        arr = (mask.astype(np.uint8) * 255)
        rgb = np.stack([arr, arr, arr], axis=-1)
        pygame.surfarray.blit_array(surf, np.transpose(rgb, (1, 0, 2)))
        pygame.image.save(surf, str(png_path))
    except Exception:
        pass  # PNG preview is a convenience only; .npy is the real artifact

    return idx, npy_path


def count_captures(out_root: Path) -> dict:
    counts = {}
    for name in set(CLASS_KEYS.values()):
        class_dir = out_root / name
        counts[name] = len(list(class_dir.glob("*.npy"))) if class_dir.exists() else 0
    return counts


def main():
    parser = argparse.ArgumentParser(description="Capture real segmented masks for shape_cnn evaluation/retraining.")
    parser.add_argument("--source", type=str, default="camera")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", type=str, choices=["color", "motion"], default="color")
    parser.add_argument("--max-dim", type=int, default=180)
    parser.add_argument("--display-scale", type=int, default=3)
    parser.add_argument("--out", type=str, default="dataset/real")
    args = parser.parse_args()

    if not PYGAME_AVAILABLE:
        print("pygame is required for this tool.")
        print(f"Import error: {PYGAME_ERROR}")
        return 1

    import pygame

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        source = make_source(source=args.source, width=args.width, height=args.height, fps=args.fps)
    except Exception as exc:
        print(f"Source error: {exc}")
        return 1

    pipeline = VisionPipeline(mode=args.mode, max_dim=args.max_dim, enable_object_info=True, use_shape_cnn=True)

    try:
        first_frame, first_ts = source.read()
        if first_frame is None:
            print("Could not read the first frame from the source.")
            return 1
        pipeline.process(first_frame, first_ts)

        pygame.init()
        pygame.display.init()
        pygame.font.init()

        H, W = pipeline.gray.shape
        scale = max(1, int(args.display_scale))
        screen = pygame.display.set_mode((int(W * scale), int(H * scale)))
        pygame.display.set_caption("mycv real-mask capture tool")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 20)

        last_saved_msg = ""
        last_saved_time = 0
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_c:
                        pipeline.set_mode("color")
                    elif event.key == pygame.K_m:
                        pipeline.set_mode("motion")
                    elif event.key == pygame.K_s:
                        pipeline.sample_color_from_center()
                    elif event.key == pygame.K_g:
                        pipeline.set_color_preset("green")
                    elif event.key == pygame.K_r:
                        pipeline.set_color_preset("red")
                    elif event.key == pygame.K_b:
                        pipeline.set_color_preset("blue")
                    elif event.key == pygame.K_y:
                        pipeline.set_color_preset("yellow")
                    else:
                        key_name = pygame.key.name(event.key)
                        if key_name in CLASS_KEYS:
                            if pipeline.selected_mask is not None and np.any(pipeline.selected_mask):
                                class_name = CLASS_KEYS[key_name]
                                idx, path = save_mask(pipeline.selected_mask, out_root, class_name)
                                last_saved_msg = f"saved {class_name} #{idx} -> {path}"
                                last_saved_time = time.perf_counter()
                                print(last_saved_msg)
                            else:
                                last_saved_msg = "nothing segmented right now -- nothing saved"
                                last_saved_time = time.perf_counter()

            frame, ts = source.read()
            if frame is None:
                print("Source returned no frame. Exiting.")
                break
            pipeline.process(frame, ts)
            if pipeline.gray is None:
                continue
            if pipeline.gray.shape != (H, W):
                H, W = pipeline.gray.shape
                screen = pygame.display.set_mode((int(W * scale), int(H * scale)))

            vis = pipeline.display_rgb.copy()
            if pipeline.mask is not None and pipeline.mask.shape == vis.shape[:2]:
                idx_mask = pipeline.mask > 0
                if np.any(idx_mask):
                    red = np.array([255, 60, 60], dtype=np.float32)
                    vis[idx_mask] = np.clip(0.45 * vis[idx_mask].astype(np.float32) + 0.55 * red, 0, 255).astype(np.uint8)

            frame_surface = pygame.image.frombuffer(np.ascontiguousarray(vis).tobytes(), (W, H), "RGB")
            display_surface = pygame.transform.scale(frame_surface, screen.get_size())
            screen.blit(display_surface, (0, 0))

            counts = count_captures(out_root)
            hud = [
                (f"mode:{pipeline.mode}  components:{len(pipeline.components)}", (255, 255, 255)),
                (f"counts: circle={counts['circle']} square={counts['square']} "
                 f"rectangle={counts['rectangle']} other={counts['other']}", (0, 255, 140)),
                ("1:circle 2:square 3:rectangle 0:other  |  c:color m:motion s:sample g/r/b/y presets  |  q:quit",
                 (190, 190, 190)),
            ]
            if last_saved_msg and time.perf_counter() - last_saved_time < 2.0:
                hud.append((last_saved_msg, (255, 255, 0)))

            y = 4
            for text, color in hud:
                surf = font.render(text, True, color)
                screen.blit(surf, (4, y))
                y += surf.get_height() + 2

            pygame.display.flip()
            clock.tick(args.fps)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        try:
            source.close()
        except Exception:
            pass
        try:
            pygame.quit()
        except Exception:
            pass

    counts = count_captures(out_root)
    print("\nFinal capture counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
