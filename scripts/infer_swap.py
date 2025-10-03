import os, glob, cv2, torch
from pathlib import Path
from src.simswap_model import DummySimSwap


SRC = 'demo/src' # 一張或多張來源臉
TGT = 'demo/tgt' # 目標照片/資料夾
OUT = 'demo/out'


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DummySimSwap().to(device)
    # 載入你最佳 checkpoint
    # model.load_state_dict(torch.load('checkpoints/ep009.pt', map_location=device))
    model.eval()


    def load_img(p):
        x = cv2.imread(p)[:,:,::-1]
        x = cv2.resize(x, (256,256))
        return torch.from_numpy(x).float().permute(2,0,1)/255.


    src_list = glob.glob(os.path.join(SRC, '*'))
    tgt_list = glob.glob(os.path.join(TGT, '*'))


    for s in src_list:
        src = load_img(s)[None].to(device)
        for t in tgt_list:
            tgt = load_img(t)[None].to(device)
            with torch.no_grad():
                out = model(src, tgt)[0].permute(1,2,0).cpu().numpy()*255
            name = f"{Path(s).stem}_TO_{Path(t).stem}.png"
            cv2.imwrite(os.path.join(OUT, name), out[:,:,::-1])
            print('saved', name)