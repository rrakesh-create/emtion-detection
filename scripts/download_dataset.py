import kagglehub
import os
import shutil

# Download the dataset (will use cache if already downloaded)
print("Ensuring dataset is downloaded...")
source_path = kagglehub.dataset_download("jettysowmith/telugu-emotion-speech")

# Define target directory in the project folder
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target_dir_name = "datasets/telugu_audio"
target_path = os.path.join(project_dir, target_dir_name)

print(f"Source path: {source_path}")
print(f"Target path: {target_path}")

# Create the target directory if it doesn't exist
if not os.path.exists(target_path):
    os.makedirs(target_path)
    print(f"Created directory: {target_path}")
else:
    print(f"Directory already exists: {target_path}")

# Copy files from source to target
print("Copying files...")
# The dataset structure from previous step seemed to have a 'telugu' subdirectory or just files.
# Let's inspect the source path contents and copy them.
# The previous output showed a 'telugu/' directory inside. 
# We probably want the contents OF that 'telugu' directory directly in 'telugu_audio', 
# or just copy the whole structure. 
# Let's copy the contents of source_path into target_path.

for item in os.listdir(source_path):
    s = os.path.join(source_path, item)
    d = os.path.join(target_path, item)
    if os.path.isdir(s):
        # If it's a directory, copy_tree behavior
        if os.path.exists(d):
            shutil.rmtree(d) # Remove if exists to ensure clean copy, or we could merge
        shutil.copytree(s, d)
    else:
        shutil.copy2(s, d)

print(f"Dataset successfully copied to: {target_path}")

# Verify
print("\nFiles in target directory:")
for root, dirs, files in os.walk(target_path):
    level = root.replace(target_path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    # Limit file listing to first 5 per directory to avoid clutter
    for f in files[:5]:
        print(f"{subindent}{f}")
    if len(files) > 5:
        print(f"{subindent}... ({len(files)-5} more files)")
