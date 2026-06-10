import torch


def augment_batch(x, edge_weight, cfg):
    # x: [B,N,W,F]
    acfg = cfg['augmentation']
    out = x.clone()
    p = float(acfg.get('sensor_dropout',0.1))
    if p > 0:
        mask = (torch.rand(out.shape[0], out.shape[1], 1, 1, device=out.device) > p).float()
        out = out * mask
    std = float(acfg.get('noise_std',0.02))
    if std > 0: out = out + torch.randn_like(out)*std
    jitter = int(acfg.get('time_jitter',0))
    if jitter > 0:
        shifts = torch.randint(-jitter, jitter+1, (out.shape[0],), device=out.device)
        for i, s in enumerate(shifts): out[i] = torch.roll(out[i], int(s.item()), dims=1)
    ew = edge_weight.clone()
    ep = float(acfg.get('edge_dropout',0.0))
    if ep > 0:
        ew = ew * (torch.rand_like(ew) > ep).float()
    return out, ew
