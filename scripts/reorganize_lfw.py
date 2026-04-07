import os, shutil
from pathlib import Path

# 🛠️ ORIEN: LFW Directory Flattener
BASE_TRAIN = Path("d:/current project/DL/dataset/face_core/train")
SOURCE_FACES = BASE_TRAIN / "lfwcrop_grey/faces"

def flatten():
    if not SOURCE_FACES.exists():
        print(f"Source doesn't exist: {SOURCE_FACES}")
        return
        
    print(f"[*] Moving subfolders from {SOURCE_FACES} to {BASE_TRAIN}")
    
    count = 0
    for folder in SOURCE_FACES.iterdir():
        if folder.is_dir():
            target = BASE_TRAIN / folder.name
            try:
                # If target exists, move files inside instead
                if target.exists():
                    for f in folder.iterdir():
                        shutil.move(str(f), str(target / f.name))
                else:
                    shutil.move(str(folder), str(target))
                count += 1
            except Exception as e:
                print(f"Error moving {folder.name}: {e}")
                
    print(f"✅ Success: Flattened {count} folders.")
    
    # Clean up empty source tree
    try:
        shutil.rmtree(BASE_TRAIN / "lfwcrop_grey")
        print(f"🗑️ Cleaned up ephemeral LFW source tree.")
    except Exception as e:
        print(f"Note: Could not delete source tree: {e}")

if __name__ == "__main__":
    flatten()
