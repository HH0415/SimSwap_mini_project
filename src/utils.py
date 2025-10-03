import os, torch
from pathlib import Path


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


class AmpScaler:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(self):
    return torch.cuda.amp.autocast(enabled=self.enabled)


def step(self, loss, optimizer):
    self.scaler.scale(loss).backward()
    self.scaler.step(optimizer)
    self.scaler.update()
    optimizer.zero_grad(set_to_none=True)