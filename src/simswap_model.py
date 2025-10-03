#（簡化包裝：載入權重、控制凍結）
import torch, torch.nn as nn


class DummySimSwap(nn.Module):
    """此處放 SimSwap 的簡化佔位（假設已有 encoder + generator）。
    實務上應引入官方實作，這裡保留接口以示範微調流程。"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,1,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
        )
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,3,3,1,1), nn.Sigmoid(),
        )


    def forward(self, src, tgt):
        # 這裡示意：以 src 的身份嵌入引導，生成 tgt 外觀（簡化）
        z = self.encoder(src)
        out = self.generator(z)
        return out


    def load_pretrained(self, path):
        # 模擬：如果你有官方 simswap state_dict，就在此載入
        pass


    def freeze_parts(self, freeze_g=True, encoder_partial=True):
        if freeze_g:
            for p in self.generator.parameters():
                p.requires_grad = False
        if encoder_partial:
            # 假設只解凍最後一層
            for i,(n,m) in enumerate(self.encoder.named_children()):
                req = (i == len(list(self.encoder.children()))-1)
                for p in m.parameters():
                    p.requires_grad = req