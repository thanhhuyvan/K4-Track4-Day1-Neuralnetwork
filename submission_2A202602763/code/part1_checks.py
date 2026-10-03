"""Part 1 diagnostics; never select a configuration using eval scores."""
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from data import prepare_data
from model import MLP,count_params,activation_stats

def run_checks(repo_root,out_dir,device):
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    data=prepare_data(device,processed_dir=str(Path(repo_root)/'data/processed'))
    means=data['X_tr'][:,:10].double().mean(0)
    stds=data['X_tr'][:,:10].double().std(0,correction=0)
    assert torch.allclose(means,torch.zeros_like(means),atol=1e-5)
    assert torch.allclose(stds,torch.ones_like(stds),atol=1e-5)
    model=MLP().to(device)
    assert count_params(model)==47879
    model.eval()
    with torch.no_grad():
        h=torch.randn(8,54,device=device)
        for layer in model.net:
            h=layer(h)
            print(type(layer).__name__,tuple(h.shape))
        assert h.shape==(8,7)
        total=0.
        for start in range(0,len(data['y_val']),8192):
            total+=F.cross_entropy(model(data['X_val'][start:start+8192]),
                                   data['y_val'][start:start+8192],reduction='sum').item()
        initial_loss=total/len(data['y_val'])
    print('Parameters:',count_params(model))
    print('Initial val loss:',initial_loss,'ln(7):',float(np.log(7)))
    print('Record actual loss; do not alter initialization to force ln(7).')
    activation=activation_stats(model,data['X_val'][:256])
    gradients={}
    model.train()
    F.cross_entropy(model(data['X_tr'][:256]),data['y_tr'][:256]).backward()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        gradients[name]=p.grad.norm().item()
        assert gradients[name]>0
        print('Gradient',name,gradients[name])
    model.zero_grad(set_to_none=True)
    # Separate model so this check cannot pretrain the baseline.
    tiny=MLP(dropout=0).to(device)
    optimizer=torch.optim.Adam(tiny.parameters(),lr=0.01)
    xb,yb=data['X_tr'][:20],data['y_tr'][:20]
    history=[]
    for step in range(1000):
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(tiny(xb),yb)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits=tiny(xb)
            current_loss=F.cross_entropy(logits,yb).item()
            acc=(logits.argmax(1)==yb).float().mean().item()
        history.append(current_loss)
        if step%100==0: print(f'Step {step+1}: loss={current_loss:.6f}, accuracy={acc:.3f}')
        if current_loss<0.01 and acc==1.: break
    out=Path(out_dir)
    for folder in ('figures','results'): (out/folder).mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots()
    ax.plot(range(1,len(history)+1),history)
    ax.set(xlabel='Update step',ylabel='Cross-entropy',title='Part 1: overfit 20 training samples')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out/'figures/health_overfit20.png',dpi=150)
    plt.show()
    plt.close(fig)
    result=dict(seed=1,parameters=count_params(model),initial_val_loss=initial_loss,
                reference_loss=float(np.log(7)),activation_std_after_relu=activation,
                gradient_norms=gradients,overfit_loss=current_loss,overfit_accuracy=acc,
                overfit_steps=len(history),overfit_history=history)
    (out/'results/health_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Overfit final:',current_loss,acc,'steps:',len(history))
    assert current_loss<0.01 and acc==1.,'Tiny batch failed; inspect diagnostics'
    print('Part 1 checks complete; diagnostics and curve saved.')
    return data,model,result
