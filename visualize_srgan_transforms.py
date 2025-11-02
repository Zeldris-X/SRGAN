import os
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


DATA_DIR = "data/srgan_generated/train"   
IMG_SIZE = 128
NUM_SAMPLES = 5                         
CLASS_NAMES = ["cat", "dog"]             


transform_srgan = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def show_augmented_samples(class_name):
    class_path = os.path.join(DATA_DIR, class_name)
    img_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
    if not img_files:
        print(f"No images found for class '{class_name}'")
        return
    
    img_path = os.path.join(class_path, random.choice(img_files))
    original = Image.open(img_path).convert("RGB")

    transformed_imgs = [transform_srgan(original).permute(1, 2, 0) for _ in range(NUM_SAMPLES)]

    
    def denorm(t):
        return (t * 0.5 + 0.5).clamp(0, 1)

    plt.figure(figsize=(15, 3))
    plt.subplot(1, NUM_SAMPLES + 1, 1)
    plt.imshow(original)
    plt.title(f"Original ({class_name})")
    plt.axis("off")

    for i, timg in enumerate(transformed_imgs, start=2):
        plt.subplot(1, NUM_SAMPLES + 1, i)
        plt.imshow(denorm(timg))
        plt.title(f"Aug {i-1}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



for cls in CLASS_NAMES:
    show_augmented_samples(cls)
