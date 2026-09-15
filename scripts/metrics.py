import mitsuba as mi
import numpy as np
import argparse
import torch

import lpips
import warnings

mi.set_variant("cuda_rgb")

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated"
)

warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated"
)

EPSILON = 1e-2

def compute_MAPE_torch(img, ref):
    e = torch.abs(img - ref) / (ref + EPSILON)
    return torch.mean(e, dim=2)

def compute_MSE_torch(img, ref):
    e = torch.square(img - ref)
    return torch.mean(e, dim=2)

def compute_relMSE_torch(img, ref):
    e = torch.square((img - ref) / (ref + EPSILON))
    return torch.mean(e, dim=2)

def compute_MAE_torch(img, ref):
    e = torch.abs(img - ref)
    return torch.mean(e, dim=2)

def compute_SMAPE_torch(img, ref):
    e = 2 * torch.abs(img - ref) / (img + ref + EPSILON)
    return torch.mean(e, dim=2)

def compute_lpips_torch(img, ref, lpips_model):
    # HDR to LDR
    exposure = 1.0
    img = torch.clamp(torch.log1p(exposure * img), 0, 1) ** (1/2.2)
    ref = torch.clamp(torch.log1p(exposure * ref), 0, 1) ** (1/2.2)
    # [H, W, 3] -> [1, 3, H, W]
    img = img.permute(2, 0, 1).unsqueeze(0)
    ref = ref.permute(2, 0, 1).unsqueeze(0)
    e = lpips_model(img, ref)
    return e.squeeze()

def compute_img_torch(img, ref, type):
    if type == "MSE":
        return compute_MSE_torch(img, ref)
    elif type == "relMSE":
        return compute_relMSE_torch(img, ref)
    elif type == "MAPE":
        return compute_MAPE_torch(img, ref)
    elif type == "MAE":
        return compute_MAE_torch(img, ref)
    elif type == "SMAPE":
        return compute_SMAPE_torch(img, ref)
    elif type == "LPIPS":
        lpips_model = lpips.LPIPS(net='alex').cuda()
        return compute_lpips_torch(img, ref, lpips_model)
    else:
        raise NotImplementedError


def compute_metric_torch(img, ref, type, discard=0.001):
    num = int(img.shape[0] * img.shape[1] * (1 - discard))
    e = compute_img_torch(img, ref, type)
    e = torch.sort(e.view(-1))[0][:num]
    return torch.mean(e)

# np.ndarray (H, W, 3) -> np.ndarray (H, W)
def compute_MSE(img, ref):
    e = np.square(img - ref)
    return np.mean(e, axis=2)

def compute_relMSE(img, ref):
    e = np.square((img - ref) / (ref + EPSILON))
    return np.mean(e, axis=2)

def compute_MAPE(img, ref):
    e = np.abs(img - ref) / (ref + EPSILON)
    return np.mean(e, axis=2)

def compute_MAE(img, ref):
    e = np.abs(img - ref)
    return np.mean(e, axis=2)

def compute_SMAPE(img, ref):
    e = 2 * np.abs(img - ref) / (img + ref + EPSILON)
    return np.mean(e, axis=2)

def compute_lpips(img, ref, lpips_model):
    # HDR to LDR
    exposure = 1.0
    img = np.clip(np.log1p(exposure * img), 0, 1)
    ref = np.clip(np.log1p(exposure * ref), 0, 1)
    # [H, W, 3] -> [1, 3, H, W]
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    ref = torch.from_numpy(ref).permute(2, 0, 1).unsqueeze(0).cuda()
    e = lpips_model(img, ref)
    return e.squeeze().detach().cpu().numpy()

def compute_img(img, ref, type):
    if type == "MSE":
        return compute_MSE(img, ref)
    elif type == "relMSE":
        return compute_relMSE(img, ref)
    elif type == "MAPE":
        return compute_MAPE(img, ref)
    elif type == "MAE":
        return compute_MAE(img, ref)
    elif type == "SMAPE":
        return compute_SMAPE(img, ref)
    elif type == "LPIPS":
        lpips_model = lpips.LPIPS(net='alex').cuda()
        return compute_lpips(img, ref, lpips_model)
    else:
        raise NotImplementedError
    
def compute_metric(img, ref, type, discard=0.001):
    num = int(img.shape[0] * img.shape[1] * (1 - discard))
    e = compute_img(img, ref, type)
    e = np.sort(e.reshape(-1))[:num]
    return np.mean(e)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compare Images")
    parser.add_argument("ref", type=str, default="ref.exr")
    parser.add_argument("img", type=str, default="hello.exr")
    parser.add_argument("metric", type=str, default="MAPE")
    
    args = parser.parse_args()
    
    img = np.nan_to_num(np.array(mi.Bitmap(args.img)), posinf=0.0, neginf=0.0)
    ref = np.nan_to_num(np.array(mi.Bitmap(args.ref)), posinf=0.0, neginf=0.0)
    
    value = compute_metric(img, ref, args.metric)
    print("{}: {:.6f}".format(args.metric, value))