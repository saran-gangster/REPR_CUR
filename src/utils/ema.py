import torch

@torch.no_grad()
def update_ema_(target_module, online_module, momentum: float = 0.996):
    for t_param, o_param in zip(target_module.parameters(), online_module.parameters()):
        if t_param.data.shape == o_param.data.shape:
            t_param.data.mul_(momentum).add_(o_param.data, alpha=(1.0 - momentum))