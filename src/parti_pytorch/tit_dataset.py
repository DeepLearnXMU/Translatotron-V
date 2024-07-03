import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image

from pathlib import Path
from shutil import rmtree
from PIL import Image
import torch
import lmdb
import pickle
import io
from torch.utils.data import Dataset, DataLoader, random_split

class TITImageDataset(Dataset):
    def __init__(
        self,
        src,
        tgt,
        folder,
        image_size,
        src_transform,
        tgt_transform = None,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.src_folder = folder + src + '/'
        self.tgt_folder = folder + tgt + '/'
        self.image_size = image_size
        self.src_paths = [p for ext in exts for p in Path(f'{self.src_folder}').glob(f'**/*.{ext}')]
        self.tgt_paths = [p for ext in exts for p in Path(f'{self.tgt_folder}').glob(f'**/*.{ext}')]

        if len(self.src_paths) != len(self.tgt_paths):
            raise Exception(f'Number of source images {len(self.src_paths)} does not match number of target images {len(self.tgt_paths)}')
        self.src_transform = src_transform
        if tgt_transform is None:
            self.tgt_transform = src_transform
        else:
            self.tgt_transform = tgt_transform

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, index):
        src_path = self.src_paths[index]
        tgt_path = self.tgt_paths[index]
        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)
        return (self.src_transform(src_img),self.tgt_transform(tgt_img))

    def select(self, indices):
        return torch.utils.data.Subset(self, indices)
    
    
class TITImageTextDataset(Dataset):
    def __init__(
        self,
        src,
        tgt,
        folder,
        image_size,
        text_folder,
        src_transform,
        tgt_transform = None,
        exts = ['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.src_folder = folder + src + '/'
        self.tgt_folder = folder + tgt + '/'
        self.image_size = image_size
        self.src_paths = [p for ext in exts for p in Path(f'{self.src_folder}').glob(f'**/*.{ext}')]
        self.tgt_paths = [p for ext in exts for p in Path(f'{self.tgt_folder}').glob(f'**/*.{ext}')]

        self.src_text_file = text_folder + "." + src
        self.tgt_text_file = text_folder + "." + tgt
        
        self.src_text = []
        self.tgt_text = []
        # read text file
        with open(self.src_text_file, 'r') as f:
            src_text = f.readlines()
            for path in self.src_paths:
                self.src_text.append(src_text[int(path.name[:path.name.find(".")])].strip())
                
        with open(self.tgt_text_file, 'r') as f:
            tgt_text = f.readlines()
            for path in self.tgt_paths:
                self.tgt_text.append(tgt_text[int(path.name[:path.name.find(".")])].strip())
        
        if len(self.src_paths) != len(self.tgt_paths):
            raise Exception(f'Number of source images {len(self.src_paths)} does not match number of target images {len(self.tgt_paths)}')

        
        self.src_transform = src_transform
        if tgt_transform is None:
            self.tgt_transform = src_transform
        else:
            self.tgt_transform = tgt_transform

    def __len__(self):
        return len(self.src_paths)

    def __getitem__(self, index):
        src_path = self.src_paths[index]
        tgt_path = self.tgt_paths[index]
        src_img = Image.open(src_path)
        tgt_img = Image.open(tgt_path)
        return (self.src_transform(src_img),self.tgt_transform(tgt_img),self.src_text[index],self.tgt_text[index])

    def select(self, indices):
        return torch.utils.data.Subset(self, indices)
    
class TITImageLmdbDataset(Dataset):
    def __init__(
        self,
        src,
        tgt,
        lmdb_path,
        image_size,
        src_transform,
        tgt_transform=None,
    ):
        super().__init__()
        src_lmdb_path = lmdb_path + src
        tgt_lmdb_path = lmdb_path + tgt
        self.src_env = lmdb.open(src_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.tgt_env = lmdb.open(tgt_lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.image_size = image_size
        self.src_transform = src_transform
        if tgt_transform is None:
            self.tgt_transform = src_transform
        else:
            self.tgt_transform = tgt_transform
        with self.src_env.begin() as src_txn, self.tgt_env.begin() as tgt_txn:
            if tgt_txn.stat()['entries'] != src_txn.stat()['entries']:
                raise Exception(f'Number of source images {src_txn.stat()["entries"]} does not match number of target images {tgt_txn.stat()["entries"]}')
            self.length = src_txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.src_env.begin() as src_txn:
            src_text, src_data = pickle.loads(src_txn.get(f"{index}".encode()))
            src_img = Image.open(io.BytesIO(src_data))
        with self.tgt_env.begin() as tgt_txn:
            tgt_text, tgt_data = pickle.loads(tgt_txn.get(f"{index}".encode()))
            tgt_img = Image.open(io.BytesIO(tgt_data))

        return (self.src_transform(src_img), self.tgt_transform(tgt_img))

    def select(self, indices):
        return torch.utils.data.Subset(self, indices)


class TITImageTextLmdbDataset(TITImageLmdbDataset):
    def __init__(
        self,
        src,
        tgt,
        lmdb_path,
        image_size,
        src_transform,
        tgt_transform=None,
    ):
        super().__init__(
            src,
            tgt,
            lmdb_path,
            image_size,
            src_transform,
            tgt_transform,
        )

    def __getitem__(self, index):
        with self.src_env.begin() as src_txn:
            src_text, src_data = pickle.loads(src_txn.get(f"{index}".encode()))
            src_img = Image.open(io.BytesIO(src_data))

        with self.tgt_env.begin() as tgt_txn:
            tgt_text, tgt_data = pickle.loads(tgt_txn.get(f"{index}".encode()))
            tgt_img = Image.open(io.BytesIO(tgt_data))

        return (self.src_transform(src_img), self.tgt_transform(tgt_img), src_text, tgt_text)


class TITImageTextMGLmdbDataset(TITImageLmdbDataset):
    def __init__(
        self,
        src,
        tgt,
        lmdb_path,
        image_size,
        src_transform,
        src_transform_mg,
        tgt_transform=None,
    ):
        super().__init__(
            src,
            tgt,
            lmdb_path,
            image_size,
            src_transform,
            tgt_transform,
        )
        self.src_transform_mg = src_transform_mg

    def __getitem__(self, index):
        with self.src_env.begin() as src_txn:
            src_text, src_data = pickle.loads(src_txn.get(f"{index}".encode()))
            src_img = Image.open(io.BytesIO(src_data))

        with self.tgt_env.begin() as tgt_txn:
            tgt_text, tgt_data = pickle.loads(tgt_txn.get(f"{index}".encode()))
            tgt_img = Image.open(io.BytesIO(tgt_data))

        return ([self.src_transform(src_img),self.src_transform_mg(src_img)], self.tgt_transform(tgt_img), src_text, tgt_text)
