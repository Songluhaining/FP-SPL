import os
import shutil

def delete_files_with_name(directory, file_name):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == file_name:
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

if __name__ == "__main__":
    folder_A = "/home/whn/codes/datasets/4wise-Elevator-FH-JML-1BUG-Full"  # 替换为文件夹A的实际路径
    file_name_to_delete = "feature_su_priors.txt"      # 替换为要删除的文件名(B文件)
    file_name_to_delete_2 = "stmt_sbfl_priors.txt"
    file_name_to_delete_3 = "cluster_0_tests.txt"
    file_name_to_delete_4 = "cluster_1_tests.txt"
    file_name_to_delete_5 = "cluster_2_tests.txt"
    if not os.path.exists(folder_A):
        print(f"Folder {folder_A} not found.")
    else:
        delete_files_with_name(folder_A, file_name_to_delete)
        delete_files_with_name(folder_A, file_name_to_delete_2)
        delete_files_with_name(folder_A, file_name_to_delete_3)
        delete_files_with_name(folder_A, file_name_to_delete_4)
        delete_files_with_name(folder_A, file_name_to_delete_5)
