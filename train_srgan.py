import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_srgan import Generator, Discriminator, VGGFeatureExtractor
from dataset_sr import get_loader


def train_srgan():
    
    CSV_PATH = "data/sr/pairs_train.csv"
    SAVE_DIR = "results/srgan"
    os.makedirs(SAVE_DIR, exist_ok=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 150              
    LR = 1e-4
    BATCH_SIZE = 8

   
    loader = get_loader(CSV_PATH, batch_size=BATCH_SIZE)
    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    vgg = VGGFeatureExtractor().to(DEVICE)

    criterion_GAN = nn.BCELoss()
    criterion_content = nn.MSELoss()
    opt_G = torch.optim.Adam(gen.parameters(), lr=LR)
    opt_D = torch.optim.Adam(disc.parameters(), lr=LR)

    
    gen_ckpt = os.path.join(SAVE_DIR, "generator_latest.pth")
    disc_ckpt = os.path.join(SAVE_DIR, "discriminator_latest.pth")
    start_epoch = 0
    if os.path.exists(gen_ckpt):
        print(f"Resuming from {gen_ckpt}")
        gen.load_state_dict(torch.load(gen_ckpt))
        if os.path.exists(disc_ckpt):
            disc.load_state_dict(torch.load(disc_ckpt))
        start_epoch = int(os.path.basename(gen_ckpt).split("_")[-1].split(".")[0]) if "epoch" in gen_ckpt else 0

    
    for epoch in range(start_epoch, EPOCHS):
        g_losses, d_losses = [], []
        for lr_imgs, hr_imgs in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            lr_imgs, hr_imgs = lr_imgs.to(DEVICE), hr_imgs.to(DEVICE)
            real = torch.ones((lr_imgs.size(0), 1), device=DEVICE)
            fake = torch.zeros((lr_imgs.size(0), 1), device=DEVICE)

            
            opt_G.zero_grad()
            sr_imgs = gen(lr_imgs)
            pred_fake = disc(sr_imgs)
            loss_GAN = criterion_GAN(pred_fake, real)
            loss_content = criterion_content(vgg(sr_imgs), vgg(hr_imgs).detach())
            loss_G = loss_content + 1e-3 * loss_GAN
            loss_G.backward()
            opt_G.step()

            
            opt_D.zero_grad()
            pred_real = disc(hr_imgs)
            pred_fake = disc(sr_imgs.detach())
            loss_D = (criterion_GAN(pred_real, real) + criterion_GAN(pred_fake, fake)) / 2
            loss_D.backward()
            opt_D.step()

            g_losses.append(loss_G.item())
            d_losses.append(loss_D.item())

        g_mean = sum(g_losses) / len(g_losses)
        d_mean = sum(d_losses) / len(d_losses)
        print(f"[Epoch {epoch+1}/{EPOCHS}] G_loss={g_mean:.4f}  D_loss={d_mean:.4f}")

        
        torch.save(gen.state_dict(), f"{SAVE_DIR}/generator_{epoch+1:03d}.pth")
        torch.save(disc.state_dict(), f"{SAVE_DIR}/discriminator_{epoch+1:03d}.pth")
        torch.save(gen.state_dict(), gen_ckpt)
        torch.save(disc.state_dict(), disc_ckpt)

        
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sr = gen(lr_imgs[:1])
                plt.figure(figsize=(9, 3))
                plt.subplot(1, 3, 1)
                plt.imshow(lr_imgs[0].permute(1, 2, 0).cpu())
                plt.title("LR"); plt.axis("off")
                plt.subplot(1, 3, 2)
                plt.imshow(sr[0].permute(1, 2, 0).cpu())
                plt.title("SR"); plt.axis("off")
                plt.subplot(1, 3, 3)
                plt.imshow(hr_imgs[0].permute(1, 2, 0).cpu())
                plt.title("HR"); plt.axis("off")
                plt.tight_layout()
                plt.savefig(f"{SAVE_DIR}/sample_{epoch+1:03d}.png")
                plt.close()


if __name__ == "__main__":
    train_srgan()
