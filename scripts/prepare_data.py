#（偵測+對齊；可選：簡單蒙版去背）
import cv2, os, glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.face_align import align_face_square
from src.utils import ensure_dir

RAW = 'data/raw'
ALIGNED = 'data/aligned'
MASKED = 'data/masked'  # 可選

if __name__ == '__main__':
    ensure_dir(ALIGNED)
    ensure_dir(MASKED)

    img_paths = []
    for ext in ('*.jpg','*.jpeg','*.png'):
        img_paths += glob.glob(os.path.join(RAW, '**', ext), recursive=True)

    for p in tqdm(img_paths):
        img = cv2.imread(p)
        if img is None:
            continue
        face256, lm = align_face_square(img, size=256)
        if face256 is None:
            continue
        rel = os.path.relpath(p, RAW)
        out_path = os.path.join(ALIGNED, Path(rel).with_suffix('.png'))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, face256)

        # 可選：極簡「臉部強化遮罩」——以人臉框中心做高斯權重，當作後續重建 loss 的 mask
        mask = np.zeros((256,256), np.uint8)
        cv2.circle(mask, (128,128), 110, 255, -1)
        mask = cv2.GaussianBlur(mask, (41,41), 0)
        mpath = os.path.join(MASKED, Path(rel).with_suffix('.png'))
        Path(mpath).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(mpath, mask)