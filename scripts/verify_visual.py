# 視覺化驗證 
import os, glob, cv2
import numpy as np
from pathlib import Path


OUT = 'demo/out'
GRID = 'demo/grid'


if __name__ == '__main__':
    os.makedirs(GRID, exist_ok=True)


    for p in glob.glob(os.path.join(OUT, '*.png')):
        out = cv2.imread(p)
        # 假設你可以組回 src/tgt：此處示意用同圖占位
        src = out.copy()
        tgt = out.copy()
        g = np.concatenate([src, tgt, out], axis=1)
        cv2.imwrite(os.path.join(GRID, Path(p).name), g)