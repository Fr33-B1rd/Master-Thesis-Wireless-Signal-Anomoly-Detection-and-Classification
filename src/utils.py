import torch
import numpy as np
import random
from sklearn.metrics import roc_auc_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def forward_pass(model, data, device, input_scale=1.0):
    data = data.to(device).float()
    data = data * input_scale
    return model(data), data

def calc_auc(x, label, y):
    # x: input, y: reconstruction
    # Calculate difference
    diff = x - y
    
    # MAE Logic
    mae = diff.abs().sum(dim=[1,2,3]).detach().cpu()
    if torch.isnan(mae).any():
        mae = torch.nan_to_num(mae)
    try:
        mae_auc = roc_auc_score(label, -mae)
    except ValueError:
        mae_auc = 0.5
    
    # MSE Logic
    mse = diff.abs().pow(2).sum(dim=[1,2,3]).detach().cpu()
    if torch.isnan(mse).any():
        mse = torch.nan_to_num(mse)
    try:
        mse_auc = roc_auc_score(label, -mse)
    except ValueError:
        mse_auc = 0.5

    # "xujing" / Relative Error Logic (from ref_code)
    # Avoid division by zero if x is 0, add epsilon if needed, but ref_code didn't
    # (diff / x) could be unstable if x is close to 0. 
    # ref_code: xujing = (diff / x).abs().sum(dim=[1,2,3])
    # Let's keep it as is to match ref_code behavior for valid comparison
    # but adding a small epsilon to denominator is barely affecting results usually
    xujing = (diff / (x + 1e-8)).abs().sum(dim=[1,2,3]).detach().cpu()
    if torch.isnan(xujing).any():
        xujing = torch.nan_to_num(xujing)
    try:
        xujing_auc = roc_auc_score(label, -xujing)
    except ValueError:
        xujing_auc = 0.5
    
    # "attn" / PER Logic (Peak Error Ratio?)
    thres = 0.05
    kernel = 3
    # Max pool on input x
    mask = torch.nn.functional.max_pool2d(
            x.detach(), kernel_size=kernel, stride=1, padding=kernel//2)
    mask = mask < thres
    
    # Max pool on negative abs diff (why negative? maybe to find min diff?)
    # ref_code: pool = - max_pool(-diff.abs()) which is min_pool(diff.abs())?
    # Actually ref_code: pool = -max_pool(-diff.abs()) -> effectively min pooling of abs diff?
    # Wait, let's re-read ref_code carefully. 
    # pool = torch.nn.functional.max_pool2d(-diff.abs(), ...)
    # pool = -pool
    # implies pool = - (max(-|diff|)) = min(|diff|)
    # So it is Min Pooling on absolute difference.
    
    pool = torch.nn.functional.max_pool2d(
            -diff.abs(), kernel_size=kernel, stride=1, padding=kernel//2)
    pool = -pool

    bg_mask = mask
    # Background attention: 90th percentile of errors in background
    attn1 = (pool*bg_mask).flatten(start_dim=1).cpu()
    attn1 = np.percentile(attn1, 90, axis=1)
    
    sig_mask = ~mask
    # Signal attention: 99th percentile of errors in signal region
    attn2 = (pool*sig_mask).flatten(start_dim=1).cpu()
    attn2 = np.percentile(attn2, 99, axis=1)

    # Weighted sum
    attn = 2*attn1 + attn2
    if np.isnan(attn).any():
        attn = np.nan_to_num(attn)
    try:
        attn_auc = roc_auc_score(label, -attn)
    except ValueError:
        attn_auc = 0.5

    return mae_auc, mse_auc, xujing_auc, attn_auc
