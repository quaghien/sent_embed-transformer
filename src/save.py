import shutil

def zip_folder(folder_path, output_path):
    shutil.make_archive(output_path, 'zip', folder_path)

# Ví dụ sử dụng
folder_path = './translation-v0-3e/'
output_path = 'full_trans-1e'
zip_folder(folder_path, output_path)