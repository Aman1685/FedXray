
import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]


class NIHChestXrayDataset(Dataset):

    def __init__(self, csv_file, image_size=224):

        self.data = pd.read_csv(csv_file)

        self.image_paths = self.data["Image Path"].tolist()

        self.labels = self.data[DISEASE_LABELS].values.astype("float32")

        self.image_names = self.data["Image Index"].tolist()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # Verify the dataset structure
        if len(self.image_paths) != len(self.labels):
            raise ValueError("Image and label counts do not match.")

        if self.labels.shape[1] != 14:
            raise ValueError("Expected exactly 14 disease labels.")

        # Check a few image paths before loading
        for path in self.image_paths[:10]:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

        print(f"Loaded: {csv_file}")
        print(f"Total images: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        # Load image as grayscale
        image = Image.open(image_path).convert("L")

        # Resize and convert to tensor
        image = self.transform(image)

        # Get 14 disease labels
        label = torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )

        return image, label
