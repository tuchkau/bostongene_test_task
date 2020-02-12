import torch
import torch.nn as nn

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    pred_area = pred.sum(dim=2).sum(dim=2)
    t_area = target.sum(dim=2).sum(dim=2)
    
    loss = (1 - ((2. * intersection + smooth) / (pred_area + t_area + smooth)))
    
    return loss.mean()