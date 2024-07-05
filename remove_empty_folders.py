import os

def remove_empty_folders(folder_path):
    """
    Recursively scans the given folder path and removes any empty folders.

    :param folder_path: The path to the folder to scan for empty folders.
    """
    # Iterate over all directories and subdirectories
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # Check and remove empty directories
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            if not os.listdir(dir_path):  # Check if the directory is empty
                os.rmdir(dir_path)
                print(f"Removed empty folder: {dir_path}")
                
remove_empty_folders("processed-videos")