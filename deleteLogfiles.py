import os
import shutil

def delete_files_with_name(directory, file_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

