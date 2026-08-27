import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'models'))
import smit
import configs_smit

config = configs_smit.get_SMIT_128_bias_True()

model = smit.SMIT_3D_Seg(config = config, out_channels = 2)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

count_parameters(model)