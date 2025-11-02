from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class SRDataset(Dataset):
    def __init__(self, csv_path, hr_size=128):
        df = pd.read_csv(csv_path)
        self.lr_paths = df["lr_path"].tolist()
        self.hr_paths = df["hr_path"].tolist()
        self.lr_size = hr_size // 4
        self.hr_size = hr_size

        self.to_tensor_lr = transforms.Compose([
            transforms.Resize((self.lr_size, self.lr_size)),
            transforms.ToTensor()
        ])
        self.to_tensor_hr = transforms.Compose([
            transforms.Resize((self.hr_size, self.hr_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.lr_paths)

    def __getitem__(self, idx):
        lr = Image.open(self.lr_paths[idx]).convert("RGB")
        hr = Image.open(self.hr_paths[idx]).convert("RGB")
        return self.to_tensor_lr(lr), self.to_tensor_hr(hr)


def get_loader(csv_path, batch_size=16):
    ds = SRDataset(csv_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
