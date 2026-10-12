"""Restore inputs and saved results from the separately supplied full package."""
from pathlib import Path
import argparse,zipfile
ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("package",type=Path,help="Path to OAF_Full_Delivery_20261005.zip")
    args=ap.parse_args()
    prefixes={"OAF_Full_Delivery_20261005/archived_complete_project/OAF_v39_1/":"project",
              "OAF_Full_Delivery_20261005/OAF_Compact_Aligned_Figures/data/":"plotting/data"}
    count=0
    with zipfile.ZipFile(args.package) as z:
        for name in z.namelist():
            if Path(name).suffix.lower() not in {".csv",".json",".npy",".npz"}:continue
            for prefix,directory in prefixes.items():
                if not name.startswith(prefix):continue
                rel=Path(name[len(prefix):])
                if rel.is_absolute() or ".." in rel.parts:raise ValueError("Unsafe archive path")
                dest=ROOT/directory/rel
                dest.parent.mkdir(parents=True,exist_ok=True)
                dest.write_bytes(z.read(name));count+=1
                break
    (ROOT/"project/verification").mkdir(parents=True,exist_ok=True)
    print(f"Restored {count} input/result files. Model fitting has not been run.")

if __name__=="__main__":main()
