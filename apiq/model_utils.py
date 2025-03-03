import os
import torch
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from squeezellm.model_parse import get_module_names, get_sequential

from apiq.quant_linear import QuantLinear, BinaryMoSLinear
from torch import nn
from peft.peft_model import PeftModelForCausalLM


def quantize_llama_like(model, weight_quant_params, quant_method: str = "default"):
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )

    for _, m in model.model.named_modules():
        if isinstance(m, (LlamaMLP, MistralMLP)):
            try:
                m.gate_proj.base_layer = QuantLinear(
                    m.gate_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.gate_proj = QuantLinear(
                    m.gate_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

            try:
                m.up_proj.base_layer = QuantLinear(
                    m.up_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.up_proj = QuantLinear(
                    m.up_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

            try:
                m.down_proj.base_layer = QuantLinear(
                    m.down_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.down_proj = QuantLinear(
                    m.down_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

        elif isinstance(m, (LlamaAttention, MistralAttention)):
            try:
                m.q_proj.base_layer = QuantLinear(
                    m.q_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.q_proj = QuantLinear(
                    m.q_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

            try:
                m.k_proj.base_layer = QuantLinear(
                    m.k_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.k_proj = QuantLinear(
                    m.k_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

            try:
                m.v_proj.base_layer = QuantLinear(
                    m.v_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.v_proj = QuantLinear(
                    m.v_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

            try:
                m.o_proj.base_layer = QuantLinear(
                    m.o_proj.base_layer, weight_quant_params=weight_quant_params, quant_method=quant_method
                )
            except:
                m.o_proj = QuantLinear(
                    m.o_proj, weight_quant_params=weight_quant_params, quant_method=quant_method
                )

    return model


def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "PeftModelForCausalLM":
        layers = model.model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def replace_modules(
    root_module, num_expert=4, zero_point_type=None, do_train=False, print_layers=False, freeze_original_weights=True
):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if (
            isinstance(module, nn.Linear) and name.find("lora") == -1
        ):  # do not quantize lora layers
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            mos_linear = BinaryMoSLinear(
                module.weight,
                module.bias,
                num_expert,
                do_train,
                zero_point_type=zero_point_type,
                freeze_original_weights=freeze_original_weights,
            )
            setattr(father, name[ind + 1 :], mos_linear)
            if print_layers:
                print(f"replace layer {name} with {mos_linear}")


def load_layer_gradients(
    gradient_dir: str, layer_idx: int, model_type: str, logging=None, device="cuda"
):
    """
    Load gradient chunk for the i-th layer, returning a dict of
    {param_name: gradient_tensor}.
    param_name must match the real parameter name in qlayer.
    """
    grad_file = os.path.join(gradient_dir, f"layer_{layer_idx}.pt")
    if not os.path.exists(grad_file):
        if logging is not None:
            logging.warning(f"Gradient file not found: {grad_file}")
        return {}

    # Read gradient file for each layer {"q": Tensor, "k": Tensor, ...}
    chunk_data = torch.load(grad_file, map_location=device)

    # Get the mapping from short keys in chunk_data to full keys in the model
    module_names = get_module_names(model_type)  # e.g. ["q", "k", "v", ...]
    seq_names = get_sequential(model_type)  # e.g. ["self_attn.q_proj", ...]

    layer_grad_dict = {}
    for short_key, full_key in zip(module_names, seq_names):
        param_key = full_key
        if short_key in chunk_data:
            layer_grad_dict[param_key] = chunk_data[short_key].clone()

    return layer_grad_dict
