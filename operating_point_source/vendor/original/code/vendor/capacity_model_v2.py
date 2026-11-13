#!/usr/bin/env python3
"""Two CNN widths, one unchanged site-patch contract. No truth enters forward."""
from __future__ import annotations
import hashlib
import random
import numpy as np
import torch
from torch import nn

PARAMETERS = {'small': 937, 'wide': 27361}


class CapacityCNN(nn.Module):
    def __init__(self, width='small'):
        super().__init__()
        if width not in PARAMETERS:
            raise ValueError('Model must be small or wide')
        a, b, h = (8, 4, 6) if width == 'small' else (64, 32, 64)
        self.conv1 = nn.Conv2d(8, a, 3, padding=1)
        self.conv2 = nn.Conv2d(a, b, 3, padding=1)
        self.hidden = nn.Linear(2*b, h)
        self.output = nn.Linear(h, 1)
        self.width = width
        if sum(p.numel() for p in self.parameters()) != PARAMETERS[width]:
            raise AssertionError('CNN parameter count changed')

    def forward(self, patches):
        if patches.ndim != 4 or tuple(patches.shape[1:]) != (8, 11, 11):
            raise ValueError('Expected float32 log1p site patches [N,8,11,11]')
        z = torch.relu(self.conv1(patches))
        z = torch.relu(self.conv2(z))
        summary = torch.cat((z.mean(dim=(2, 3)), z[:, :, 5, 5]), dim=1)
        return self.output(torch.relu(self.hidden(summary))).squeeze(-1)


def seed_all(seed, threads=2):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.set_num_threads(int(threads))
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False


def create_model(width, seed, device='cpu', threads=2):
    # Initialize on CPU, as in the archived CNN. Move only after initialization.
    seed_all(seed, threads)
    return CapacityCNN(width).to(device)


def parameter_hash(model):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def tensor_patches(patches, device='cpu'):
    a = np.asarray(patches)
    if a.dtype != np.uint8 or a.ndim != 4 or tuple(a.shape[1:]) != (8, 11, 11):
        raise ValueError('Expected uint8 site patches [N,8,11,11]')
    if a.size and int(a.max()) > 25:
        raise ValueError('Site-patch count bound exceeded')
    # Preserve conversion order. No normalization is fitted to validation data.
    return torch.log1p(torch.from_numpy(np.array(a, copy=True, order='C')).to(
        device=device, dtype=torch.float32))
