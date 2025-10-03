import os, glob, cv2, numpy as np, pandas as pd
from src.arcface_embedder import ArcFaceONNX
from skimage.metrics import structural_similarity as ssim
import lpips


OUT = 'demo/out'
ARC = 'weights/arcface_r100.onnx'


if __name__ == '__main__':
    arc = ArcFaceONNX(ARC)
    lp = lpips.LPIPS(net='alex')


    rows = []
    for p in glob.glob(os.path.join(OUT, '*.png')):
        img = cv2.imread(p)[:,:,::-1]
        # 推導對應的 src/tgt 名（根據 infer 命名規則）
        bname = os.path.basename(p)
        src_name, tgt_name = bname.split('_TO_')
        tgt_name = tgt_name.replace('.png','')


        # 這裡示意：若能回推到實際 src,tgt 原圖，可載入做 SSIM/LPIPS（此處簡化用 output 自身當 placeholder）
        out = img
        # 身份相似（以輸出 vs 假定源臉，示意）：
        f_out = arc(out)[0]
        f_src = f_out # *實務中請改成 arc(對應 src 圖)
        cos = float(np.dot(f_out, f_src))


        # 影像品質（若有對應 target 才有意義）：
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        s = ssim(gray, gray) # placeholder
        l = float(lp(torch.tensor(out/255.).permute(2,0,1)[None]*2-1,
                        torch.tensor(out/255.).permute(2,0,1)[None]*2-1))


        rows.append({'name': bname, 'ID_cosine': cos, 'SSIM': s, 'LPIPS': l})


    df = pd.DataFrame(rows)
    print(df.describe())
    df.to_csv('demo/metrics_summary.csv', index=False)
    print('Saved demo/metrics_summary.csv')