import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.utils.data as data

from typing import List
import numpy as np
from tqdm.auto import tqdm


class ImageDataset(data.Dataset):
    def __init__(self, imgs):
        super().__init__()
        self.mytransform = self.input_transform()
        self.images = imgs

    def __getitem__(self, index):
        img = self.images[index]
        img = self.mytransform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    @staticmethod
    def input_transform():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(480),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])


class EigenPlacesFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        if torch.cuda.is_available():
            print('Using GPU')
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            print('Using MPS')
            self.device = torch.device("mps")
        else:
            print('Using CPU')
            self.device = torch.device("cpu")
        self.model = torch.hub.load("gmberton/eigenplaces", "get_trained_model",
                                    backbone="ResNet50", fc_output_dim=2048)
        self.dim = 2048
        self.preprocess = ImageDataset.input_transform()
        self.model = self.model.to(self.device)
        self.model.eval()

    def _compute_features_small_batch(self, imgs: List[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.empty((0, self.dim), dtype=np.float32)

        batch = torch.stack([self.preprocess(img) for img in imgs], dim=0)
        batch = batch.to(self.device, non_blocking=self.device.type == "cuda")
        with torch.inference_mode():
            image_encoding = self.model(batch)
        return image_encoding.detach().cpu().numpy().astype(np.float32, copy=False)

    def compute_features(self, imgs: List[np.ndarray]) -> np.ndarray:
        if not imgs:
            return np.empty((0, self.dim), dtype=np.float32)

        if len(imgs) <= 8:
            return self._compute_features_small_batch(imgs)

        img_set = ImageDataset(imgs)
        num_workers = 0 if len(img_set) < 32 else min(4, os.cpu_count() or 1)
        batch_size = min(8, len(img_set))
        test_data_loader = DataLoader(
            dataset=img_set,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=self.device.type == "cuda",
        )
        show_progress = len(img_set) >= batch_size * 4
        iterator = tqdm(test_data_loader) if show_progress else test_data_loader
        with torch.inference_mode():
            global_feats = np.empty((len(img_set), self.dim), dtype=np.float32)
            for input_data, indices in iterator:
                indices_np = indices.numpy()
                input_data = input_data.to(self.device, non_blocking=self.device.type == "cuda")
                image_encoding = self.model(input_data)
                global_feats[indices_np, :] = image_encoding.cpu().numpy()
        return global_feats

