import os
from os.path import join
import argparse
import cv2
from tqdm import tqdm
import numpy as np

DATASET_PATHS = {
    "original": "original_sequences",
    "DeepFakeDetection": "manipulated_sequences/DeepFakeDetection",
    "DeepFakes": "manipulated_sequences/DeepFakes",
    "Face2Face": "manipulated_sequences/Face2Face",
    "FaceSwap": "manipulated_sequences/FaceSwap",
}
COMPRESSION = ["c0", "c23", "c40"]
OUTPUT_SIZES = [(150, 150), (196, 196), (224, 224)]  # Additional sizes


def extract_and_process_frames(
    data_path, mask_path, output_path, method="cv2", padding=40, extra_top=40  # 20,35
):
    """Extracts the 19th frame of each second from a video, applies mask, crops with extra top area, resizes, and saves images."""
    os.makedirs(output_path, exist_ok=True)

    if method == "cv2":
        video_reader = cv2.VideoCapture(data_path)
        mask_reader = cv2.VideoCapture(mask_path)
        fps = video_reader.get(cv2.CAP_PROP_FPS)  # Get frames per second

        if fps == 0:
            raise ValueError("Unable to determine FPS for the video.")

        target_frame = int(fps * 19 / 20)  # Calculate 19th frame index
        frame_count = 0
        saved_frame_num = 0

        while video_reader.isOpened() and mask_reader.isOpened():
            success_video, frame = video_reader.read()
            success_mask, mask = mask_reader.read()

            if not success_video or not success_mask:
                break

            # Ensure the mask is single-channel and matches frame size
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

            # Save only the 19th frame of each second
            if frame_count % int(fps) == target_frame:
                # Find bounding box around the non-black (masked) area
                non_zero_pixels = cv2.findNonZero(mask)
                x, y, w, h = cv2.boundingRect(non_zero_pixels)  # Bounding box

                # Adjust bounding box to include more top area
                x = max(0, x - padding)
                y = max(0, y - padding - extra_top)  # Extend upwards
                w = min(frame.shape[1] - x, w + 2 * padding)
                h = min(frame.shape[0] - y, h + 2 * padding + extra_top)

                # Crop the face region from the original manipulated frame using the adjusted bounding box
                cropped_frame = frame[y : y + h, x : x + w]

                # Save the original resolution cropped face image with extended top area
                original_folder = join(output_path, "original")
                os.makedirs(original_folder, exist_ok=True)
                cv2.imwrite(
                    join(original_folder, f"{saved_frame_num:04d}.png"), cropped_frame
                )

                # Save resized images in different dimensions
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
    else:
        raise Exception("Invalid extraction method: {}".format(method))


def extract_videos_with_masks(data_path, mask_root_path, dataset, compression):
    """Extracts the 19th frame each second from all videos and their masks for a given dataset and compression."""
    videos_path = join(data_path, DATASET_PATHS[dataset], compression, "videos")
    images_path = join(data_path, DATASET_PATHS[dataset], compression, "images_1")
    masks_path = join(mask_root_path)

    for video in tqdm(os.listdir(videos_path)):
        video_name, _ = os.path.splitext(video)
        mask_file = f"{video_name}.mp4"  # Assumes masks follow the same naming pattern as videos

        video_path = join(videos_path, video)
        mask_path = join(masks_path, mask_file)
        output_folder = join(images_path, video_name)

        if os.path.exists(mask_path):
            extract_and_process_frames(video_path, mask_path, output_folder)
        else:
            print(f"Mask for video {video} not found!")


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
        "--dataset",
        "-d",
        type=str,
        choices=list(DATASET_PATHS.keys()) + ["all"],
        default="all",
        help="Specify dataset type or 'all' for all datasets.",
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

    if args.dataset == "all":
        for dataset in DATASET_PATHS.keys():
            extract_videos_with_masks(
                args.data_path, args.mask_path, dataset, args.compression
            )
    else:
        extract_videos_with_masks(
            args.data_path, args.mask_path, args.dataset, args.compression
        )
