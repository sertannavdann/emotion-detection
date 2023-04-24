import os
path_train = 'data/merged/train/'
path_test = 'data/merged/test/'
path_balance = 'data/balanced/'

def rename_files(path):
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            count = 1
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                new_file_name = f"{folder}_{count}.jpg"
                new_file_path = os.path.join(folder_path, new_file_name)
                os.rename(file_path, new_file_path)
                count += 1

rename_files(path_train)
rename_files(path_test)
#rename_files(path_balance)