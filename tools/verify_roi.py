"""
ROI visualization tool for verifying polygon alignment.

Overlays ROI and stopline polygons on camera frames to visually verify
that the regions are correctly positioned over the approach lane.

Usage:
    python tools/verify_roi.py --tl_dir outputs/Town10HD_Opt/tl_road5_lane-1_s10058
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import cv2
import numpy as np
import glob
from pathlib import Path


def load_and_visualize(tl_dir, save_output=True):
    """Load ROI/stopline polygons and overlay on first frame."""
    print("="*80)
    print("ROI VISUALIZATION TOOL")
    print("="*80)

    tl_path = Path(tl_dir)

    # Load ROI polygon
    roi_file = tl_path / "roi.json"
    if not roi_file.exists():
        print(f"✗ ROI file not found: {roi_file}")
        return False

    try:
        with open(roi_file, 'r') as f:
            roi_data = json.load(f)
        roi_polygon = np.array(roi_data["polygon"], dtype=np.int32)
        print(f"✓ Loaded ROI polygon: {len(roi_polygon)} vertices")
    except Exception as e:
        print(f"✗ Failed to load ROI: {e}")
        return False

    # Load stopline polygon (optional)
    stopline_file = tl_path / "stopline.json"
    stopline_polygon = None
    if stopline_file.exists():
        try:
            with open(stopline_file, 'r') as f:
                stopline_data = json.load(f)
            stopline_polygon = np.array(stopline_data["polygon"], dtype=np.int32)
            print(f"✓ Loaded stopline polygon: {len(stopline_polygon)} vertices")
        except Exception as e:
            print(f"⚠ Stopline file exists but failed to load: {e}")
    else:
        print(f"⚠ Stopline file not found (will be derived from ROI)")

    # Find first frame image
    frame_pattern = str(tl_path / "frame_*.png")
    frames = sorted(glob.glob(frame_pattern))

    if not frames:
        print(f"✗ No frame images found matching: {frame_pattern}")
        print(f"  Note: Frames are only saved when SAVE_EVERY_N > 0 in config.py")
        return False

    first_frame = frames[0]
    print(f"✓ Found {len(frames)} frames, using: {os.path.basename(first_frame)}")

    # Load image
    img = cv2.imread(first_frame)
    if img is None:
        print(f"✗ Failed to load image: {first_frame}")
        return False

    print(f"✓ Image loaded: {img.shape[1]}x{img.shape[0]} (WxH)")

    # Create overlay
    overlay = img.copy()

    # Draw ROI polygon (yellow, thick)
    cv2.polylines(overlay, [roi_polygon.reshape((-1, 1, 2))], True, (0, 255, 255), 3)
    cv2.fillPoly(overlay, [roi_polygon.reshape((-1, 1, 2))], (0, 255, 255))

    # Draw stopline polygon (red, thin) if available
    if stopline_polygon is not None:
        cv2.polylines(overlay, [stopline_polygon.reshape((-1, 1, 2))], True, (0, 0, 255), 2)
        cv2.fillPoly(overlay, [stopline_polygon.reshape((-1, 1, 2))], (0, 0, 255))

    # Blend overlay with original (30% opacity)
    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Add labels
    cv2.putText(result, "ROI (yellow)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, cv2.LINE_AA)
    if stopline_polygon is not None:
        cv2.putText(result, "Stopline (red)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Save output
    if save_output:
        output_file = tl_path / "roi_overlay.png"
        cv2.imwrite(str(output_file), result)
        print(f"✓ Saved overlay: {output_file}")

    # Display (optional - may not work in headless environments)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("ROI Overlay Verification")
        plt.axis('off')

        plot_file = tl_path / "roi_overlay_plot.png"
        plt.savefig(str(plot_file), bbox_inches='tight', dpi=150)
        print(f"✓ Saved plot: {plot_file}")

        # Try to show (may fail in headless)
        try:
            plt.show()
        except Exception:
            print(f"  (Display skipped - no GUI available)")

        plt.close()

    except ImportError:
        print(f"⚠ matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"⚠ Plot generation failed: {e}")

    print("\n" + "="*80)
    print("VERIFICATION CHECKLIST")
    print("="*80)
    print("1. Does the yellow ROI polygon cover the approach lane?")
    print("2. Is the red stopline positioned near the intersection stop bar?")
    print("3. Are the polygons axis-aligned or slightly rotated (expected)?")
    print("4. Do the regions overlap with where vehicles will queue?")
    print("\n✓ If YES to all → ROI is correctly positioned")
    print("✗ If NO to any → ROI needs adjustment (rerun setup with correct coordinates)")
    print("="*80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize ROI polygon overlay")
    parser.add_argument("--tl_dir", type=str, required=True,
                        help="Traffic light directory containing roi.json and frames")
    parser.add_argument("--no_save", action="store_true",
                        help="Don't save output images")
    args = parser.parse_args()

    success = load_and_visualize(args.tl_dir, save_output=not args.no_save)

    if success:
        print("\n✓ ROI visualization complete")
    else:
        print("\n✗ ROI visualization failed")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
