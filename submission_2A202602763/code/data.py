"""Fit numeric scaling on the training subset only."""
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
N_NUMERIC=10
def load_split(processed_dir='data/processed'):
    with np.load(Path(processed_dir)/'train.npz') as f: X,y=f['X'],f['y']
    with np.load(Path(processed_dir)/'eval.npz') as f: Xe,ye,ids=f['X'],f['y'],f['row_id']
    for features,labels in ((X,y),(Xe,ye)):
        assert features.shape==(len(labels),54) and labels.ndim==1
        assert features.dtype==np.float32 and labels.dtype==np.int64
        assert np.all((labels>=0)&(labels<7))
    return X,y,Xe,ye,ids
def make_val_split(X,y,val_fraction=0.2,seed=42):
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=val_fraction,stratify=y,random_state=seed)
    return Xtr,ytr,Xv,yv
def fit_standardizer(X_tr):
    mean=X_tr[:,:10].mean(0,dtype=np.float64)
    std=X_tr[:,:10].std(0,dtype=np.float64)
    return mean,np.where(std==0,1.,std)
def apply_standardizer(X,mean,std):
    result=X.copy()
    result[:,:10]=(result[:,:10]-mean)/np.where(std==0,1.,std)
    assert np.isfinite(result).all()
    return result
def prepare_data(device,val_fraction=0.2,seed=42,processed_dir='data/processed'):
    X,y,Xe,ye,ids=load_split(processed_dir)
    Xtr,ytr,Xv,yv=make_val_split(X,y,val_fraction,seed)
    mean,std=fit_standardizer(Xtr)
    result={name:torch.as_tensor(apply_standardizer(values,mean,std),device=device)
            for name,values in (('X_tr',Xtr),('X_val',Xv),('X_eval',Xe))}
    result.update({name:torch.as_tensor(values,device=device)
                   for name,values in (('y_tr',ytr),('y_val',yv),('y_eval',ye))})
    result.update(eval_row_id=ids,mean=mean,std=std)
    for name in ('X_tr','X_val','X_eval'): print(name,tuple(result[name].shape))
    majority=np.bincount(ytr,minlength=7).argmax()
    print('Majority accuracy on val:',float((yv==majority).mean()))
    return result
def iterate_batches(X,y,batch_size,generator=None,shuffle=True):
    if batch_size<=0 or len(X)!=len(y): raise ValueError('Invalid batch size or lengths')
    index_device=generator.device if generator is not None else X.device
    indices=(torch.randperm(len(X),generator=generator,device=index_device).to(X.device)
             if shuffle else torch.arange(len(X),device=X.device))
    for start in range(0,len(X),batch_size):
        idx=indices[start:start+batch_size]
        yield X[idx],y[idx]
