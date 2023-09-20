import torch.optim as optim
from optimizers import *

def get_optimizer(optim_name, model_parm, lr, weight_decay):
    if optim_name == 'SGD':
        optimizer = optim.SGD(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'Adagrad':
        optimizer = optim.Adagrad(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'Adam':
        optimizer = optim.Adam(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'AdamW':
        optimizer = optim.AdamW(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'LARS':
        optimizer = LARS(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'LAMB':
        optimizer = LAMB(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'Lion':
        optimizer = Lion(model_parm, lr=lr, weight_decay=weight_decay)
    return optimizer

def get_bnb_optimizer(optim_name, model_parm, lr, weight_decay):
    import bitsandbytes as bnb

    if optim_name == 'SGD':
        optimizer = bnb.optim.SGD(model_parm, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_name == 'Adagrad':
        optimizer = bnb.optim.Adagrad(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'Adam':
        optimizer = bnb.optim.PagedAdam(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'AdamW':
        optimizer = bnb.optim.PagedAdamW(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'LARS':
        optimizer = bnb.optim.LARS(model_parm, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim_name == 'LAMB':
        optimizer = bnb.optim.LAMB(model_parm, lr=lr, weight_decay=weight_decay)
    elif optim_name == 'Lion':
        optimizer = bnb.optim.PagedLion(model_parm, lr=lr, weight_decay=weight_decay)
    return optimizer