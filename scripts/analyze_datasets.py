"""
Full Dataset Analysis + Cleanup Report
Identifies all files, duplicates, archives that are safe to remove
"""
import os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = r"D:\current project\DL\dataset"

def human_size(n):
    for u in ['B','KB','MB','GB']:
        if abs(n) < 1024.0: return f"{n:3.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} TB"

def scan_dir(path):
    total_size = 0
    file_count = 0
    dir_count  = 0
    ext_counts = {}
    for root, dirs, files in os.walk(path):
        dir_count += len(dirs)
        for f in files:
            fp = os.path.join(root, f)
            try:
                sz = os.path.getsize(fp)
            except:
                sz = 0
            total_size += sz
            file_count += 1
            ext = os.path.splitext(f)[1].lower() or "(no ext)"
            ext_counts[ext] = ext_counts.get(ext, {"count":0,"size":0})
            ext_counts[ext]["count"] += 1
            ext_counts[ext]["size"]  += sz
    return total_size, file_count, dir_count, ext_counts

report = []
grand_total = 0

print("\n" + "="*70)
print("  ORIEN DATASET FULL ANALYSIS")
print("="*70)

modalities = sorted(os.listdir(ROOT))
for mod in modalities:
    mod_path = os.path.join(ROOT, mod)
    if not os.path.isdir(mod_path): continue

    size, files, dirs, exts = scan_dir(mod_path)
    grand_total += size

    print(f"\n{'─'*70}")
    print(f"  [{mod.upper()}]  {human_size(size)}  |  {files} files  |  {dirs} sub-dirs")
    print(f"{'─'*70}")

    # List top-level items
    for item in sorted(os.listdir(mod_path)):
        ipath = os.path.join(mod_path, item)
        if os.path.isdir(ipath):
            # Count contents
            cnt = sum(len(fs) for _, _, fs in os.walk(ipath))
            sz = sum(os.path.getsize(os.path.join(r,f))
                     for r,_,fs in os.walk(ipath) for f in fs
                     if os.path.exists(os.path.join(r,f)))
            print(f"    [DIR ] {item:<45}  {cnt:>5} files  {human_size(sz):>10}")
        else:
            sz = os.path.getsize(ipath)
            print(f"    [FILE] {item:<45}  {1:>5} files  {human_size(sz):>10}")

    # Extension breakdown
    print(f"\n    Extension breakdown:")
    for ext, info in sorted(exts.items(), key=lambda x: -x[1]['size']):
        print(f"      {ext:<12}  {info['count']:>5} files  {human_size(info['size']):>10}")

    report.append({"mod": mod, "size": size, "files": files, "dirs": dirs, "exts": exts})

print(f"\n{'='*70}")
print(f"  GRAND TOTAL: {human_size(grand_total)}")
print(f"{'='*70}")

# ── IDENTIFY CLEANUP CANDIDATES ──────────────────────────────────────
print(f"\n{'='*70}")
print(f"  CLEANUP ANALYSIS — Files Safe to Remove")
print(f"{'='*70}")

cleanup = []

for mod in modalities:
    mod_path = os.path.join(ROOT, mod)
    if not os.path.isdir(mod_path): continue

    for root, dirs, files in os.walk(mod_path):
        for f in files:
            fp = os.path.join(root, f)
            rel = fp.replace(ROOT + os.sep, "")
            try: sz = os.path.getsize(fp)
            except: sz = 0

            reason = None

            # Rule 1: archive.zip/.tar.gz when extracted content exists
            if f in ("archive.zip", "archive.tar.gz"):
                # Check if the modality has extracted content (dirs other than archive itself)
                parent = os.path.dirname(fp)
                sibling_dirs = [d for d in os.listdir(parent) if os.path.isdir(os.path.join(parent, d))]
                sibling_files = [x for x in os.listdir(parent) if x != f and os.path.isfile(os.path.join(parent, x))]
                if sibling_dirs or sibling_files:
                    reason = "Archive already extracted — extracted content exists"

            # Rule 2: Duplicate directories (lfwcrop_grey = same content as face/faces + face/lists)
            if "lfwcrop_grey" in rel:
                reason = "Duplicate — content lifted to face/faces/ and face/lists/"

            # Rule 3: Recognition-wrapper (repo wrapper, actual data in subjects/)
            if "Recognition-main" in rel:
                # Check if subjects/ exists
                subjects = os.path.join(ROOT, "face_orl", "subjects")
                if os.path.isdir(subjects):
                    reason = "Duplicate — extracted to face_orl/subjects/"

            # Rule 4: gesture-source-v1 (same as gesture/classes/)
            if "gesture-150k" in rel:
                classes = os.path.join(ROOT, "gesture", "classes")
                if os.path.isdir(classes):
                    reason = "Duplicate — content copied to gesture/classes/"

            if reason:
                cleanup.append({"path": fp, "rel": rel, "size": sz, "reason": reason})

# Group by mod
by_mod = {}
for item in cleanup:
    top = item["rel"].split(os.sep)[0]
    by_mod.setdefault(top, []).append(item)

total_savings = 0
for mod, items in sorted(by_mod.items()):
    mod_savings = sum(i["size"] for i in items)
    total_savings += mod_savings
    print(f"\n  [{mod.upper()}]  potential savings: {human_size(mod_savings)}")
    for i in items:
        print(f"    REMOVE: {i['rel'][:65]}")
        print(f"            ({human_size(i['size'])}) -- {i['reason']}")

# Walk and report full dirs to remove
print(f"\n  Directories to remove entirely:")
dirs_to_remove = []
# lfwcrop_grey inside face/
lg = os.path.join(ROOT, "face", "lfwcrop_grey")
if os.path.isdir(lg):
    sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(lg) for f in fs)
    print(f"    REMOVE DIR: face\\lfwcrop_grey\\  ({human_size(sz)}) -- duplicated in face/faces/")
    dirs_to_remove.append(lg); total_savings += sz

# Inner folder
orl_inner = os.path.join(ROOT, "face_orl", "Recognition-main")
if os.path.isdir(orl_inner):
    sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(orl_inner) for f in fs)
    print(f"    REMOVE DIR: face_orl\\Recognition-main\\  ({human_size(sz)}) -- extracted to subjects/")
    dirs_to_remove.append(orl_inner); total_savings += sz

# HAGRID original folder inside gesture/
hagrid = os.path.join(ROOT, "gesture", "hagrid-classification-512p-no-gesture-150k")
if os.path.isdir(hagrid):
    sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(hagrid) for f in fs)
    print(f"    REMOVE DIR: gesture\\hagrid-...\\  ({human_size(sz)}) -- copied to gesture/classes/")
    dirs_to_remove.append(hagrid); total_savings += sz

print(f"\n{'─'*70}")
print(f"  TOTAL RECLAIMABLE: {human_size(total_savings)}")
print(f"{'─'*70}")
print("\n  Run cleanup script: python scripts/cleanup_datasets.py")
