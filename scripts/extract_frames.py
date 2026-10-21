"""Extract evenly spaced frames from a video for PatchCore datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import IMAGE_EXTENSIONS, PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Directory for extracted JPEG frames.")
    parser.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="Save one frame at this interval in seconds. Use 0 to save every frame (default: 1).",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0, help="Start extracting at this video timestamp (default: 0).")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Length of the extraction segment; 0 means until the end (default: 0).")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional maximum number of saved frames; 0 means unlimited.")
    parser.add_argument("--clear", action="store_true", help="Delete existing images in the output directory before extraction.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video).expanduser()
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if args.every_seconds < 0:
        raise ValueError("--every-seconds must be 0 or greater.")
    if args.start_seconds < 0 or args.duration_seconds < 0:
        raise ValueError("--start-seconds and --duration-seconds must be 0 or greater.")
    if args.max_frames < 0:
        raise ValueError("--max-frames must be 0 or greater.")

    output_dir = Path(args.output).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clear:
        for file_path in output_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                file_path.unlink()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1 if args.every_seconds == 0 else max(1, round(fps * args.every_seconds))
    start_frame = round(args.start_seconds * fps)
    end_frame = start_frame + round(args.duration_seconds * fps) if args.duration_seconds > 0 else None

    frame_index = 0
    frames_read = 0
    saved_count = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frames_read += 1
            current_index = frame_index
            frame_index += 1

            if current_index < start_frame:
                continue
            if end_frame is not None and current_index >= end_frame:
                break

            if (current_index - start_frame) % frame_interval == 0:
                output_path = output_dir / f"{saved_count:06d}.jpg"
                if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"Could not save frame: {output_path}")
                saved_count += 1
                if args.max_frames and saved_count >= args.max_frames:
                    break

            if frame_index and frame_index % 500 == 0:
                suffix = f"/{total_frames}" if total_frames > 0 else ""
                print(f"Processed {frame_index}{suffix} frames; saved {saved_count}.")
    finally:
        capture.release()

    print("Frame extraction complete")
    print(f"Video       : {video_path}")
    print(f"Frames read : {frames_read}")
    print(f"Segment      : start={args.start_seconds:.2f}s, duration={args.duration_seconds:.2f}s (0 means to end)")
    print(f"Frames saved: {saved_count}")
    print(f"Output      : {output_dir}")


if __name__ == "__main__":
    main()
