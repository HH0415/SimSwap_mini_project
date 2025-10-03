import cv2, torch
import pandas as pd
from torch.utils.data import Dataset


class PairSwapDataset(Dataset):
    def __init__(self, csv_path, mask_dir=None):
        self.df = pd.read_csv(csv_path)
        self.mask_dir = mask_dir


def __len__(self):
    return len(self.df)


def _read(self, p):
    img = cv2.imread(p)[:,:,::-1]
    return torch.from_numpy(img).float().permute(2,0,1)/255.


def __getitem__(self, idx):
    row = self.df.iloc[idx]
    src = self._read(row.src_path)
    tgt = self._read(row.tgt_path)
    return {
        'src': src, 'tgt': tgt,
        'src_path': row.src_path, 'tgt_path': row.tgt_path,
}