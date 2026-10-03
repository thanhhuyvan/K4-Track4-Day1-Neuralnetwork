"""MLP for the three architectures allowed by the lab."""
import torch
from torch import nn
EXPECTED_PARAMS = {(256,128):47879, (512,256):161287, (256,128,64):55687}
class MLP(nn.Module):
    def __init__(self, hidden=(256,128), dropout=0., init='he', in_features=54, num_classes=7):
        super().__init__()
        if tuple(hidden) not in EXPECTED_PARAMS or not 0 <= dropout < 1:
            raise ValueError('Invalid architecture or dropout')
        layers=[]
        for width in hidden:
            layers.extend([nn.Linear(in_features,width),nn.ReLU()])
            if dropout: layers.append(nn.Dropout(dropout))
            in_features=width
        layers.append(nn.Linear(in_features,num_classes))
        self.net=nn.Sequential(*layers)
        init_weights(self,init)
    def forward(self,x):
        return self.net(x)
def init_weights(model,init):
    if init not in {'default','he','xavier','zeros','normal'}:
        raise ValueError(init)
    if init=='default': return
    for layer in model.modules():
        if isinstance(layer,nn.Linear):
            if init=='he': nn.init.kaiming_normal_(layer.weight,nonlinearity='relu')
            elif init=='xavier': nn.init.xavier_normal_(layer.weight)
            elif init=='zeros': nn.init.zeros_(layer.weight)
            else: nn.init.normal_(layer.weight,mean=0,std=0.01)
            nn.init.zeros_(layer.bias)
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
@torch.no_grad()
def activation_stats(model,x):
    """Standard deviation after each hidden ReLU; preserve model mode."""
    training=model.training
    model.eval()
    stats=[]
    try:
        for layer in model.net:
            x=layer(x)
            if isinstance(layer,nn.ReLU): stats.append(x.std().item())
    finally:
        model.train(training)
    return stats
