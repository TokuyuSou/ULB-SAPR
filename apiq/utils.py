from math import inf
from collections import OrderedDict
import torch
from torch import nn
from apiq.quant_linear import BaseQuantLinear, QuantLinear, BinaryMoSLinear

METHOD_TO_KEYS = {
    "default": ["bound_factor"],
    "DB-LLM": ["alpha"],
}

def get_learnable_parameters_from_class(module: nn.Module, class_name: str, name: bool = False):
    """
    Retrieves all learnable parameters (requires_grad=True) of a specific class type within a given module.

    Args:
        module (nn.Module): The parent module to search in.
        class_name (str): The name of the class whose learnable parameters are to be retrieved.
        name (bool): If True, return the names of the parameters. If False, return the parameters themselves.

    Returns:
        List[nn.Parameter]: A list of learnable parameters with requires_grad=True.
    """
    learnable_params = []
    for _, submodule in module.named_modules():
        if submodule.__class__.__name__ == class_name:
            # Filter parameters with requires_grad=True
            if name:
                learnable_params.extend(
                    n for n, p in submodule.named_parameters() if p.requires_grad
                )
            else:
                learnable_params.extend(
                    p for p in submodule.parameters() if p.requires_grad
                )
    return iter(learnable_params)


def quant_temporary(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True


def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

@torch.no_grad()   
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight)
            module.use_temporary_parameter=False

def add_new_module(name, original_module, added_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = original_module
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], added_module)
    else:
        setattr(original_module, name, added_module)


def get_named_linears(module):
    return {
        name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)
    }

def register_scales_and_zeros(model):
    for _, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

def set_quant_state(self, weight_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    for n, m in self.named_modules():
        if isinstance(m, BaseQuantLinear):
            m.set_quant_state(weight_quant)

def get_lwc_parameters(model, name=False):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            if name:
                params.append(n)
            else:
                params.append(m)
    return iter(params)

def get_peft_parameters(model, peft_method):
    if peft_method == "LoRA" or peft_method == "DoRA":
        key = "lora"
    else:
        raise ValueError("Only support LoRA and DoRA for now")
    params = []
    for n, m in model.named_parameters():
        if n.find(key) > -1:
            params.append(m)
    return iter(params)


def get_quantization_parameters(model, quant_method, name=False):
    if quant_method == "BinaryMoS":
        return get_learnable_parameters_from_class(model, "BinaryMoSLinear", name)
    elif quant_method in METHOD_TO_KEYS:
        keys = METHOD_TO_KEYS[quant_method]
        params = []
        for n, m in model.named_parameters():
            if any(key in n for key in keys):
                if name:
                    params.append(n)
                else:
                    params.append(m)
        return iter(params)
    else:
        raise ValueError(f"Quantization method {quant_method} not supported")


def get_all_learnable_parameters(model, quant_method, peft_method):
    if quant_method == "BinaryMoS":
        return get_binary_mos_parameters(model, peft_method)
    elif quant_method in METHOD_TO_KEYS:
        peft_params = get_peft_parameters(model, peft_method)
        quantization_params = get_quantization_parameters(model, quant_method)
        return iter(list(peft_params) + list(quantization_params))
    else:
        raise ValueError(f"Quantization method {quant_method} not supported")


def get_apiq_parameters(model, peft_method):
    if peft_method == "LoRA" or peft_method == "DoRA":
        key = "lora"
    else:
        raise ValueError("Only support LoRA and DoRA for now")
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(key) > -1:
            params.append(m)
    return iter(params)


def get_binary_mos_parameters(model, peft_method):
    if peft_method == "LoRA" or peft_method == "DoRA":
        key = "lora"
    else:
        raise ValueError("Only support LoRA and DoRA for now")
    peft_params = get_peft_parameters(model, peft_method)
    quantization_params = get_learnable_parameters_from_class(model, "BinaryMoSLinear")
    return iter(list(peft_params) + list(quantization_params))

def lwc_state_dict(model, destination=None, prefix='', keep_vars=False, quant_method="default"):
    keys = METHOD_TO_KEYS[quant_method]
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if any(key in name for key in keys):
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def peft_state_dict(model, peft_method, destination=None, prefix='', keep_vars=False):
    if peft_method == "LoRA" or peft_method == "DoRA":
        key = "lora"
    else:
        raise ValueError("Only support LoRA and DoRA for now")
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find(key) > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

@torch.no_grad()
def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)