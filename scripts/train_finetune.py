import os, yaml, torch, lpips
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm
from src.datasets import PairSwapDataset
from src.simswap_model import DummySimSwap
from src.arcface_embedder import ArcFaceONNX
from src.utils import AmpScaler


L1 = nn.L1Loss()


class Perceptual:
    def __init__(self):
        self.fn = lpips.LPIPS(net='alex').cuda()
    def __call__(self, a, b):
        return self.fn(a*2-1, b*2-1).mean()




def cosine_id_loss(arc, src_rgb, out_rgb):
    # src/out: (B,3,256,256) -> numpy(H,W,3)
    loss = 0.0
    for i in range(src_rgb.size(0)):
        s = (src_rgb[i].permute(1,2,0).detach().cpu().numpy()*255).astype('uint8')
        o = (out_rgb[i].permute(1,2,0).detach().cpu().numpy()*255).astype('uint8')
        fs = arc(s)[0]
        fo = arc(o)[0]
        sim = ArcFaceONNX.cosine(fs, fo)
        loss += (1.0 - sim)
    return loss / src_rgb.size(0)


if __name__ == '__main__':
    cfg = yaml.safe_load(open('configs/train_256_amp.yaml','r'))


    train_ds = PairSwapDataset(cfg['train_pairs_csv'])
    val_ds = PairSwapDataset(cfg['val_pairs_csv'])


    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
    num_workers=cfg['num_workers'], pin_memory=True)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DummySimSwap().to(device)
# 實務：model.load_pretrained(cfg['weights']['simswap_pretrained'])
# 實務：model.freeze_parts(**cfg['freeze'])


if cfg['optimizer']['type'].lower() == 'adam':
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=cfg['optimizer']['lr'], betas=tuple(cfg['optimizer']['betas']))


scaler = AmpScaler(enabled=cfg['amp'])
perc = Perceptual()
arc = ArcFaceONNX(cfg['weights']['arcface'])

global_step = 0

for epoch in range(cfg['max_epochs']):
    model.train()
    pbar = tqdm(train_loader, ncols=100, desc=f"ep{epoch}")
    _accu = 0
    optimizer.zero_grad(set_to_none=True)


    for step, batch in enumerate(pbar):
        src = batch['src'].to(device, non_blocking=True)
        tgt = batch['tgt'].to(device, non_blocking=True)


        with scaler.autocast():
            out = model(src, tgt)
            rec = L1(out, tgt)
            per = perc(out, tgt)
            idl = cosine_id_loss(arc, src, out)
            loss = cfg['loss']['rec_coef']*rec + cfg['loss']['perc_coef']*per + cfg['loss']['id_coef']*idl


        scaler.scaler.scale(loss / cfg['grad_accum_steps']).backward()
        _accu += 1
        if _accu % cfg['grad_accum_steps'] == 0:
            scaler.scaler.step(optimizer)
            scaler.scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1


        pbar.set_postfix({
            'rec': float(rec.item()),
            'perc': float(per.item()),
            'id': float(idl),
            'loss': float(loss.item())
        })


# TODO：保存 checkpoint、簡易驗證
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), f'checkpoints/ep{epoch:03d}.pt')
    print(f"[Saved] checkpoints/ep{epoch:03d}.pt")