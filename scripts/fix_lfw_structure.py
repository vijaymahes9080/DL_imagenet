import os, shutil
from pathlib import Path

# 🛠️ ORIEN: LFW Neural Data Scaffolder [FIX]
DATA_PATH = Path("d:/current project/DL/dataset/face_core/train/lfwcrop_grey/faces")

def fix_lfw():
    if not DATA_PATH.exists():
        print(f"Path not found: {DATA_PATH}")
        return

    files = list(DATA_PATH.glob("*.jpg"))
    print(f"[*] Processing {len(files)} files in {DATA_PATH}")
    
    count = 0
    for f in files:
        name = f.stem # e.g. AJ_Cook_0001
        # Extract name part (everything before the last underscore and digits)
        # Typically "FirstName_LastName_0001" or "FirstName_0001"
        try:
            # Find the last underscore
            last_underscore = name.rfind('_')
            class_name = name[:last_underscore]
            
            class_dir = DATA_PATH / class_name
            class_dir.mkdir(exist_ok=True)
            
            shutil.move(str(f), str(class_dir / f.name))
            count += 1
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            
    print(f"✅ Success: Moved {count} images into class folders.")

if __name__ == "__main__":
    fix_lfw()
