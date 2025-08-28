import os
from os.path import join
import argparse
import cv2
from tqdm import tqdm
import re
import numpy as np
from mtcnn import MTCNN

# Paths to directories in the dataset
DATASET_PATHS = {"DeepFakeDetection": "original_sequences/actors"}
COMPRESSION = ["c0", "c23", "c40"]
OUTPUT_SIZES = [(150, 150), (196, 196), (224, 224)]


def parse_mask_filename(mask_filename):
    match = re.match(r"(\d+)_(\d+)__([a-zA-Z0-9_]+)__[A-Z0-9]{8}\.mp4", mask_filename)
    if match:
        target_actor = match.group(1)
        sequence_name = match.group(3)
        return target_actor, sequence_name
    return None, None


def extract_and_process_frames_with_mtcnn(
    video_path, output_path, padding=40, extra_top=40
):
    """Fallback function using MTCNN to detect and extract only one face when mask is unavailable or durations mismatch."""
    video_reader = cv2.VideoCapture(video_path)
    detector = MTCNN()

    os.makedirs(output_path, exist_ok=True)

    saved_frame_num = 0
    frame_count = 0
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 19 / 20)  # 19 FPS target frame index

    while video_reader.isOpened():
        success, frame = video_reader.read()
        if not success:
            break

        # Only process the frame if it matches the target frame count
        if frame_count % int(fps) == target_frame:
            # Run MTCNN on each frame
            detections = detector.detect_faces(frame)

            if detections:
                # Select the detection with the highest confidence
                best_detection = max(detections, key=lambda x: x["confidence"])

                x, y, w, h = best_detection["box"]
                # x = max(0, x)
                # y = max(0, y)

                # cropped_frame = frame[y : y + h, x : x + w]

                x = max(0, x - padding)
                y = max(0, y - padding - extra_top)
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding + extra_top)

                cropped_frame = frame[y : y + h, x : x + w]

                # Save original cropped face
                original_folder = join(output_path, "original")
                os.makedirs(original_folder, exist_ok=True)
                cv2.imwrite(
                    join(original_folder, f"{saved_frame_num:04d}.png"), cropped_frame
                )

                # Save resized images
                for size in OUTPUT_SIZES:
                    resized_frame = cv2.resize(cropped_frame, size)
                    size_folder = join(output_path, f"{size[0]}x{size[1]}")
                    os.makedirs(size_folder, exist_ok=True)
                    cv2.imwrite(
                        join(size_folder, f"{saved_frame_num:04d}.png"), resized_frame
                    )

                saved_frame_num += 1

        frame_count += 1

    video_reader.release()


def extract_and_process_frames(
    data_path, mask_path, output_path, method="cv2", padding=40, extra_top=40
):
    video_reader = cv2.VideoCapture(data_path)
    mask_reader = cv2.VideoCapture(mask_path)

    duration_video = (
        video_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        / video_reader.get(cv2.CAP_PROP_FPS)
        * 1000
    )
    duration_mask = (
        mask_reader.get(cv2.CAP_PROP_FRAME_COUNT)
        / mask_reader.get(cv2.CAP_PROP_FPS)
        * 1000
    )

    # If durations don't match, switch to MTCNN fallback
    if abs(duration_video - duration_mask) > 1000:
        print(f"Duration mismatch. Using MTCNN for face extraction: {data_path}")
        video_reader.release()
        mask_reader.release()
        extract_and_process_frames_with_mtcnn(data_path, output_path)
        return

    os.makedirs(output_path, exist_ok=True)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 19 / 20)
    frame_count = 0
    saved_frame_num = 0

    while video_reader.isOpened() and mask_reader.isOpened():
        success_video, frame = video_reader.read()
        success_mask, mask = mask_reader.read()

        if not success_video or not success_mask:
            break

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        if frame_count % int(fps) == target_frame:
            non_zero_pixels = cv2.findNonZero(mask)
            x, y, w, h = cv2.boundingRect(non_zero_pixels)
            x = max(0, x - padding)
            y = max(0, y - padding - extra_top)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding + extra_top)

            cropped_frame = frame[y : y + h, x : x + w]

            original_folder = join(output_path, "original")
            os.makedirs(original_folder, exist_ok=True)
            cv2.imwrite(
                join(original_folder, f"{saved_frame_num:04d}.png"), cropped_frame
            )

            for size in OUTPUT_SIZES:
                resized_frame = cv2.resize(cropped_frame, size)
                size_folder = join(output_path, f"{size[0]}x{size[1]}")
                os.makedirs(size_folder, exist_ok=True)
                cv2.imwrite(
                    join(size_folder, f"{saved_frame_num:04d}.png"), resized_frame
                )

            saved_frame_num += 1

        frame_count += 1

    video_reader.release()
    mask_reader.release()


def extract_original_videos_with_masks(data_path, mask_root_path, compression):
    original_videos_path = join(
        data_path, DATASET_PATHS["DeepFakeDetection"], compression, "videos"
    )
    masks_path = join(mask_root_path)
    images_output_path = join(
        data_path, DATASET_PATHS["DeepFakeDetection"], compression, "images_1"
    )

    for video in tqdm(os.listdir(original_videos_path)):
        video_name, _ = os.path.splitext(video)
        match = re.match(r"(\d+)__([a-zA-Z0-9_]+)", video_name)
        if not match:
            print(f"Skipping file with unrecognized format: {video}")
            continue

        actor_number, scene_name = match.groups()

        matching_mask = None
        for mask_video in os.listdir(masks_path):
            target_actor, mask_scene_name = parse_mask_filename(mask_video)
            if target_actor == actor_number and mask_scene_name == scene_name:
                matching_mask = mask_video
                break

        if matching_mask:
            video_path = join(original_videos_path, video)
            mask_path = join(masks_path, matching_mask)
            output_folder = join(images_output_path, video_name)
            extract_and_process_frames(video_path, mask_path, output_folder)
        else:
            print(f"No mask found for original video, using MTCNN: {video}")
            video_path = join(original_videos_path, video)
            output_folder = join(images_output_path, video_name)
            extract_and_process_frames_with_mtcnn(video_path, output_folder)


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data_path", type=str, required=True, help="Path to dataset root directory."
    )
    p.add_argument(
        "--mask_path",
        type=str,
        required=True,
        help="Path to root directory containing mask videos.",
    )
    p.add_argument(
        "--compression",
        "-c",
        type=str,
        choices=COMPRESSION,
        default="c0",
        help="Compression level.",
    )
    args = p.parse_args()

    extract_original_videos_with_masks(args.data_path, args.mask_path, args.compression)
