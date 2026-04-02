import os
import shutil

repo_root = r"c:\Users\kqnau\OneDrive\Desktop\College\Semester_Project\Traffic-speed-prediction"
os.chdir(repo_root)

# Create folders
folders_to_create = [
    "Not_Needed/root",
    "Not_Needed/MT-STGIN",
    "Not_Needed/3S-TBLN",
    "Not_Needed/ST-ANet"
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

# Files to move from root
root_files = [".DS_Store", "temp.py", "training.log", "training_output.txt"]
for file in root_files:
    if os.path.exists(file):
        try:
            shutil.move(file, os.path.join("Not_Needed/root", file))
            print(f"Moved {file}")
        except Exception as e:
            print(f"Failed to move {file}: {e}")

# Files to move from MT-STGIN
mt_files = [
    ".DS_Store", "debug_batch.py", "debug_format.py", "debug_minute.py", 
    "debug_train.py", "debug_output.txt", "temp_debug.txt", "training_output.txt",
    "check_device.py", "check_gpu.py", "quick_test.py", "test_data_load.py",
    "test_generator.py", "test_metr_la.py", "test_simplified_data.py", 
    "test_training_step.py", "train_launcher.py", "train_metr_la.py", 
    "train_with_metr_la.py"
]

for file in mt_files:
    path = os.path.join("MT-STGIN", file)
    if os.path.exists(path):
        try:
            shutil.move(path, os.path.join("Not_Needed/MT-STGIN", file))
            print(f"Moved {path}")
        except Exception as e:
            print(f"Failed to move {path}: {e}")

# Move from 3S-TBLN
s3_files = [".DS_Store", "Response of Review Comments"]
for file in s3_files:
    path = os.path.join("3S-TBLN", file)
    if os.path.exists(path):
        try:
            shutil.move(path, os.path.join("Not_Needed/3S-TBLN", file))
            print(f"Moved {path}")
        except Exception as e:
            print(f"Failed to move {path}: {e}")

# Move from ST-ANet
st_files = [".DS_Store"]
for file in st_files:
    path = os.path.join("ST-ANet", file)
    if os.path.exists(path):
        try:
            shutil.move(path, os.path.join("Not_Needed/ST-ANet", file))
            print(f"Moved {path}")
        except Exception as e:
            print(f"Failed to move {path}: {e}")

# Pycache clean up
for root, dirs, files in os.walk(repo_root):
    if "Not_Needed" in root:
        continue
    for d in dirs:
        if d == "__pycache__":
            pycache_path = os.path.join(root, d)
            try:
                shutil.rmtree(pycache_path)
                print(f"Removed {pycache_path}")
            except Exception as e:
                pass

print("Cleanup complete!")
