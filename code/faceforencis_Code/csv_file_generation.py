import os
import pandas as pd
import re


def get_image_count(folder_path, extensions=(".png", ".jpg", ".jpeg", ".gif")):
    """Count images in a folder with various extensions"""
    folder_path = os.path.normpath(os.path.join(folder_path, "original"))
    # print(folder_path)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0
    return len(
        [
            file
            for file in os.listdir(folder_path)
            if any(file.lower().endswith(ext) for ext in extensions)
        ]
    )


def generate_folder_mapping_csv(manipulated_dir, original_dir, output_csv):
    data = []
    matched_folders = []

    # Process manipulated images folders
    for manipulated_folder in os.listdir(manipulated_dir):
        manipulated_folder_path = os.path.join(manipulated_dir, manipulated_folder)

        if os.path.isdir(manipulated_folder_path):
            # More precise parsing of the manipulated folder name
            match = re.match(
                r"(\d+)_(\d+)__([a-zA-Z0-9_]+)__[A-Z0-9]{8}$", manipulated_folder
            )

            if match:
                target_actor = match.group(1)  # First number (27 in your example)
                scene_identifier = match.group(3)  # Scene name (kitchen_pan)

                # Construct the expected original folder name
                expected_original_folder = f"{target_actor}__{scene_identifier}"

                # Find matching original folders
                matching_originals = [
                    orig_folder
                    for orig_folder in os.listdir(original_dir)
                    if os.path.isdir(os.path.join(original_dir, orig_folder))
                    and orig_folder == expected_original_folder
                ]

                if matching_originals:
                    for original_folder_name in matching_originals:
                        original_folder_path = os.path.join(
                            original_dir, original_folder_name
                        )

                        manipulated_count = get_image_count(manipulated_folder_path)
                        original_count = get_image_count(original_folder_path)

                        data.append(
                            [
                                manipulated_folder,  # Original manipulated folder name
                                original_folder_name,  # Matched original folder name
                                manipulated_count,  # Number of images in manipulated folder
                                original_count,  # Number of images in original folder
                            ]
                        )

                        matched_folders.append(
                            {
                                "manipulated": manipulated_folder,
                                "original": original_folder_name,
                            }
                        )
                else:
                    print(
                        f"No matching original folder found for: {manipulated_folder}"
                    )
                    print(f"Expected original folder: {expected_original_folder}")
            else:
                print(f"Could not parse folder name: {manipulated_folder}")

    # Save data to CSV with more informative print
    df = pd.DataFrame(
        data,
        columns=[
            "manipulated_folder",
            "original_folder",
            "manipulated_image_count",
            "original_image_count",
        ],
    )

    df.to_csv(output_csv, index=False)
    print(f"CSV saved to {output_csv}")
    print(f"Total matched folders: {len(df)}")

    # Detailed matching report
    print("\nDetailed Folder Matching:")
    for match in matched_folders:
        print(f"Manipulated: {match['manipulated']} ← Original: {match['original']}")


# Example usage
generate_folder_mapping_csv(
    r"D:\Mtech\sem1\ML\deep fake\data\manipulated_sequences\DeepFakeDetection\c23\images",
    r"D:\Mtech\sem1\ML\deep fake\data\original_sequences\actors\c23\images_1",
    "../../data/folder_image_counts.csv",
)
