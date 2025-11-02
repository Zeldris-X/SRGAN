import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from model_srgan import Generator   



LR_ROOT = Path("data/sr")           
OUT_ROOT = Path("data/srgan_generated")
MODEL_PATH = Path("results/srgan/generator_150.pth")   
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPSCALE = 4
HR_SIZE = 128


def load_model():
    gen = Generator().to(DEVICE)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    gen.eval()
    print(f"Loaded generator: {MODEL_PATH}")
    return gen


def process_split(split, gen):
    src_dir = LR_ROOT / split / "LR_x4"
    out_dir = OUT_ROOT / split
    transform = transforms.Compose([
        transforms.Resize((HR_SIZE // UPSCALE, HR_SIZE // UPSCALE)),
        transforms.ToTensor(),
    ])
    to_pil = transforms.ToPILImage()

    for class_dir in sorted(src_dir.glob("*")):
        if not class_dir.is_dir():
            continue
        cls = class_dir.name
        out_cls = out_dir / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        imgs = list(class_dir.glob("*.*"))
        for img_path in tqdm(imgs, desc=f"{split}/{cls}", ncols=80):
            try:
                img = Image.open(img_path).convert("RGB")
                lr_tensor = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    sr_tensor = gen(lr_tensor)
                sr_img = to_pil(sr_tensor.squeeze().cpu().clamp(0, 1))
                sr_img.save(out_cls / f"{img_path.stem}_SR.png")
            except Exception as e:
                print(f"Failed {img_path}: {e}")


def main():
    gen = load_model()
    for split in ["train", "test"]:
        process_split(split, gen)
    print(f"\nAll SR images saved to → {OUT_ROOT}")


if __name__ == "__main__":
    main()
