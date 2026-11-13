"""Graph-only model creation; reproducible CPU initialization and CUDA transfer."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
from capacity_model_v2 import seed_all,parameter_hash
from graph_models_v1 import GeometricMPNN
VARIANTS=('full2','full1','no_xy2','current2')
PARAMETERS={v:sum(p.numel() for p in GeometricMPNN(v).parameters()) for v in VARIANTS}
def create_model(width,seed,device='cpu',threads=2):
    seed_all(seed,threads);m=GeometricMPNN(width).to(device);m.variant=width;return m
