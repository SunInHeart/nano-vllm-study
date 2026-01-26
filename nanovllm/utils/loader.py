import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_moudles_mapping = getattr(model, "packed_moudles_mapping", {})
    for file in glob(os.path.join(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # print(f"{weight_name}, {f.get_tensor(weight_name).shape}")
                # such as: model.layers.0.self_attn.q_proj.weight | torch.size(2048, 1024)
                for k in packed_moudles_mapping:
                    if k in weight_name:
                        v, shard_id = packed_moudles_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else: # for loop not break, which means current weight_name not match all map keys
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
