import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd


SRC_ROOT = Path(r"C:\Rashed\8119\split")    
DST_ROOT = Path("data/sr")         


HR_SIZE = 128
SCALE = 4
LR_SIZE = HR_SIZE // SCALE 

def list_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def save_pair(src_path, hr_path, lr_path):
    img = Image.open(src_path).convert("RGB")
    
    hr_img = img.resize((HR_SIZE, HR_SIZE), Image.BICUBIC)
    lr_img = hr_img.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)

    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)

    
    hr_img.save(hr_path, format="PNG", optimize=True)
    lr_img.save(lr_path, format="PNG", optimize=True)

def process_split(split):
    src_split = SRC_ROOT / split
    if not src_split.exists():
        print(f"[WARN] {src_split} not found. Skipping.")
        return pd.DataFrame(columns=["lr_path","hr_path","class","split"])

    pairs = []
    
    for cls_dir in sorted([p for p in src_split.iterdir() if p.is_dir()]):
        cls_name = cls_dir.name
        imgs = list_images(cls_dir)
        for p in tqdm(imgs, desc=f"{split}/{cls_name}"):
            rel = p.relative_to(src_split / cls_name)
           
            hr_out = DST_ROOT / split / "HR" / cls_name / (p.stem + ".png")
            lr_out = DST_ROOT / split / "LR_x4" / cls_name / (p.stem + ".png")

            save_pair(p, hr_out, lr_out)
            pairs.append({
                "lr_path": str(lr_out.as_posix()),
                "hr_path": str(hr_out.as_posix()),
                "class": cls_name,
                "split": split
            })

    df = pd.DataFrame(pairs)
    df_out = DST_ROOT / f"pairs_{split}.csv"
    df.to_csv(df_out, index=False)
    print(f"[OK] Wrote {len(df)} pairs → {df_out}")
    return df

def main():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    df_train = process_split("train")
    df_test  = process_split("test")

    
    for df, name in [(df_train,"train"), (df_test,"test")]:
        if df is not None and len(df):
            print(f"{name}: classes =", sorted(df['class'].unique()))
            print(df.head())

if __name__ == "__main__":
    main()
