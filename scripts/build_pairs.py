#（建立 source→target 配對，用於 supervision & identity）
import os, glob, random, pandas as pd
from pathlib import Path


ALIGNED = 'data/aligned'


random.seed(34021)


def list_images(root):
    paths = []
    for ext in ('*.jpg','*.jpeg','*.png'):
        paths += glob.glob(os.path.join(root, '**', ext), recursive=True)
    return paths


if __name__ == '__main__':
    imgs = list_images(ALIGNED)
    # 假設以「資料夾名」當作 identity（id01/xxx.png）
    pairs = []
    id_to_imgs = {}
    for p in imgs:
        ident = Path(p).parts[-2]
        id_to_imgs.setdefault(ident, []).append(p)


id_list = list(id_to_imgs.keys())
# 生成跨人配對（source_id != target_id）
for sid in id_list:
    for s in id_to_imgs[sid]:
        tid = random.choice([x for x in id_list if x!=sid])
        t = random.choice(id_to_imgs[tid])
        pairs.append([sid, tid, s, t])


random.shuffle(pairs)
n = len(pairs)
split = int(n*0.9)
train = pairs[:split]
val = pairs[split:]


pd.DataFrame(train, columns=['src_id','tgt_id','src_path','tgt_path']).to_csv('data/train_pairs.csv', index=False)
pd.DataFrame(val, columns=['src_id','tgt_id','src_path','tgt_path']).to_csv('data/val_pairs.csv', index=False)
print(f"train: {len(train)}, val: {len(val)}")