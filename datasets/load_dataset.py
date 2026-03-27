#   =====================================================================
#   Copyright (C) 2023  Stefan Schubert, stefan.schubert@etit.tu-chemnitz.de
#
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <http://www.gnu.org/licenses/>.
#   =====================================================================
#
import os
import urllib.request
import zipfile
from glob import glob
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from typing import List, Tuple
from abc import ABC, abstractmethod
import re


class Dataset(ABC):
    @abstractmethod
    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def download(self, destination: str):
        pass


class GardensPointDataset(Dataset):
    def __init__(self, destination: str = 'images/GardensPoint/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset GardensPoint day_right--night_right')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'day_right/*.jpg'))
        fns_q = sorted(glob(self.destination + 'night_right/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        GThard = np.eye(len(imgs_db)).astype('bool')
        GTsoft = convolve2d(GThard.astype('int'),
                            np.ones((17, 1), 'int'), mode='same').astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== GardensPoint dataset does not exist. Download to ' + destination + '...')

        fn = 'GardensPoint_Walking.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class StLuciaDataset(Dataset):
    def __init__(self, destination: str = 'images/StLucia_small/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset StLucia 100909_0845--180809_1545 (small version)')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + '100909_0845/*.jpg'))
        fns_q = sorted(glob(self.destination + '180809_1545/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== StLucia dataset does not exist. Download to ' + destination + '...')

        fn = 'StLucia_small.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)


class SFUDataset(Dataset):
    def __init__(self, destination: str = 'images/SFU/'):
        self.destination = destination

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        print('===== Load dataset SFU dry--jan')

        # download images if necessary
        if not os.path.exists(self.destination):
            self.download(self.destination)

        # load images
        fns_db = sorted(glob(self.destination + 'dry/*.jpg'))
        fns_q = sorted(glob(self.destination + 'jan/*.jpg'))

        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # create ground truth
        gt_data = np.load(self.destination + 'GT.npz')
        GThard = gt_data['GThard'].astype('bool')
        GTsoft = gt_data['GTsoft'].astype('bool')

        return imgs_db, imgs_q, GThard, GTsoft

    def download(self, destination: str):
        print('===== SFU dataset does not exist. Download to ' + destination + '...')

        fn = 'SFU.zip'
        url = 'https://www.tu-chemnitz.de/etit/proaut/datasets/' + fn

        # create folders
        path = os.path.expanduser(destination)
        os.makedirs(path, exist_ok=True)

        # download
        urllib.request.urlretrieve(url, path + fn)

        # unzip
        with zipfile.ZipFile(path + fn, 'r') as zip_ref:
            zip_ref.extractall(destination)

        # remove zipfile
        os.remove(destination + fn)

class CampusDataset(Dataset):
    """
    Loader for custom campus dataset with day/night images.

    Dataset structure:
    - custom_dataset/day_images/: Reference database (day images)
    - custom_dataset/night_images/: Query images (night images)

    Naming convention:
    - Matching pairs: image042.jpg (night) matches image042.jpg (day)
    - No perfect match: image043-npm.jpg (night) has no match in day set
    - Additional no-match files: npm01.jpg, npm02.jpg, PXL_*.jpg
    """

    def __init__(self, destination: str = 'custom_dataset/'):
        self.destination = destination
        self.day_dir = os.path.join(destination, 'day_images')
        self.night_dir = os.path.join(destination, 'night_images')

    def load(self) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        Load campus dataset images and generate ground truth.

        Returns:
            imgs_db: List of database (day) images as numpy arrays
            imgs_q: List of query (night) images as numpy arrays
            GThard: Binary ground truth matrix [N_db x N_q]
            GTsoft: Dilated ground truth matrix [N_db x N_q]
        """
        print('===== Load custom campus dataset: day_images --> night_images')

        if not os.path.exists(self.day_dir):
            raise FileNotFoundError(f"Day images directory not found: {self.day_dir}")
        if not os.path.exists(self.night_dir):
            raise FileNotFoundError(f"Night images directory not found: {self.night_dir}")

        # Load day images (database/reference)
        fns_db = sorted(glob(os.path.join(self.day_dir, '*.jpg')))
        imgs_db = [np.array(Image.open(fn)) for fn in fns_db]

        # Extract day image numbers for matching
        day_names = [os.path.splitext(os.path.basename(fn))[0] for fn in fns_db]
        print(f"  Loaded {len(imgs_db)} day images (database)")

        # Load night images (queries)
        fns_q = sorted(glob(os.path.join(self.night_dir, '*.jpg')))
        imgs_q = [np.array(Image.open(fn)) for fn in fns_q]

        # Extract night image numbers for matching
        night_names = [os.path.splitext(os.path.basename(fn))[0] for fn in fns_q]
        print(f"  Loaded {len(imgs_q)} night images (queries)")

        # Create ground truth matrix
        GThard = self._create_ground_truth(day_names, night_names)

        # Create soft ground truth (dilated version - allows nearby matches)
        # Dilate by ±2 positions in the database
        GTsoft = convolve2d(GThard.astype('int'),
                           np.ones((5, 1), 'int'), mode='same').astype('bool')

        # Print matching statistics
        n_matches = np.sum(GThard.any(axis=0))
        n_no_match = len(imgs_q) - n_matches
        print(f"  Ground truth: {n_matches} queries have matches, {n_no_match} have no match (-npm)")

        return imgs_db, imgs_q, GThard, GTsoft

    def _create_ground_truth(self, day_names: List[str], night_names: List[str]) -> np.ndarray:
        """
        Create ground truth matrix based on filename matching.

        Matching rules:
        - image042.jpg (night) matches image042.jpg (day)
        - image043-npm.jpg (night) has no match
        - npm01.jpg, PXL_*.jpg have no match

        Args:
            day_names: List of day image names (without extension)
            night_names: List of night image names (without extension)

        Returns:
            GThard: Binary matrix [N_db x N_q] where GThard[i,j]=1 means
                   day image i matches night image j
        """
        N_db = len(day_names)
        N_q = len(night_names)
        GThard = np.zeros((N_db, N_q), dtype=bool)

        for j, night_name in enumerate(night_names):
            # Skip images marked as no-perfect-match
            if '-npm' in night_name.lower() or night_name.startswith('npm'):
                continue

            # Skip Pixel phone images (no corresponding day images)
            if night_name.startswith('PXL_'):
                continue

            # Extract base image number (e.g., "image042" from "image042.jpg")
            # This handles cases like "image012 (2).jpg" as well
            match = re.match(r'(image\d+)', night_name)
            if match:
                base_name = match.group(1)

                # Find matching day image
                if base_name in day_names:
                    i = day_names.index(base_name)
                    GThard[i, j] = True

        return GThard

    def download(self, destination: str):
        """CampusDataset is already local, no download needed."""
        raise NotImplementedError("CampusDataset should already exist locally in custom_dataset/")
