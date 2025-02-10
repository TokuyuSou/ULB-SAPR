import os
import sys
import shutil
import argparse
import logging
import random
import json
import time
import numpy as np
from pathlib import Path
from collections import OrderedDict

import torch
import peft
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.mistral.modeling_mistral import MistralPreTrainedModel

from apiq.model_utils import quantize_llama_like, get_blocks, replace_modules
from apiq.data_utils import get_loaders
from apiq.calibrate import calibrate
from apiq.evaluate import evaluate
from apiq.quant_linear import QuantLinear


logging.basicConfig(
    filename=os.path.join(
        "/home/ubuntu/ApiQ/ApiQ/logs", f"{time.strftime('%Y-%m-%d-%H-%M-%S')}_apiq.log"
    ),
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


MODEL_FAMILY = ["llama", "mistral"]


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Initialization
    args.model_family = args.model_name_or_path.split("/")[-1].split("-")[0].lower()
    assert (
        args.model_family in MODEL_FAMILY
    ), f"Currently don't support {args.model_family}"
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        # Create subfolder for each quantization method and run (based on time)
        args.save_dir = os.path.join(args.save_dir, args.quant_method, "run_" + time.strftime('%Y-%m-%d-%H-%M-%S'))
        print(f"Save dir: {args.save_dir}")
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if args.convert_to_gptq:
        assert (args.resume is not None) and (
            args.resume != args.save_dir
        ), "--resume refers to the folder of fake quant."
    assert not (
        args.epochs > 1 and args.convert_to_gptq
    ), "--convert_to_gptq can only be set after calibration"

    # Load model and tokenizer
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, attn_implementation=args.attn_implementation
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=False, legacy=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        device_map=args.device,
        torch_dtype=torch.float16,
    )
    assert (
        args.seqlen <= config.max_position_embeddings
    ), "The sequence length of calibration samples exceed the model's"

    weight_quant_params = {
        "n_bits": args.wbits,
        "symmetric": args.symmetric,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point,
    }

    peft_config_kwargs = json.loads(args.peft_args)
    if args.peft_method == "LoRA":
        # peft_config_kwargs["lora_alpha"] = 16 if args.wbits == 4 else peft_config_kwargs["r"] # borrowed from LoftQ
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=target_modules,
            **peft_config_kwargs,
        )
        model = peft.get_peft_model(model, peft_config)
    elif args.peft_method == "DoRA":
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
        peft_config_kwargs["use_dora"] = True
        peft_config = peft.LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            target_modules=target_modules,
            **peft_config_kwargs,
        )
        model = peft.get_peft_model(model, peft_config)

    assert isinstance(
        model.base_model.model, (LlamaPreTrainedModel, MistralPreTrainedModel)
    )

    if args.quant_method == "BinaryMoS":
        replace_modules(
            get_blocks(model),
            num_expert=args.num_expert,
            zero_point_type=args.zero_point_type,
            do_train=True,
            print_layers=True,
            freeze_original_weights=False if args.train_original_weights else True
        )
    else:
        model = quantize_llama_like(model, weight_quant_params, quant_method=args.quant_method)
    model.eval()
    logging.info(model)

    # Quantization
    logging.info("=== start quantization ===")
    tick = time.time()
    cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_name_or_path.split("/")[-1]}_{args.calib_dataset}_n{args.nsamples}len{args.seqlen}.cache'
    if os.path.exists(cache_dataloader):
        dataloader = torch.load(cache_dataloader)
        logging.info(f"load calibration data from {cache_dataloader}")
    else:
        dataloader, _ = get_loaders(
            args.calib_dataset,
            tokenizer,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=args.seqlen,
        )
        torch.save(dataloader, cache_dataloader)

    calibrate(model, args, dataloader, logging=logging)
    logging.info(f"Time for quantization: {time.time() - tick} s")
    evaluate(model, tokenizer, args, logging)

    logging.info(
        f"Save fake quant model, i.e. the quant weight is in fp16. For real quant model, use --convert_to_gptq after quantization."
    )
    model.save_pretrained(
        os.path.join(args.save_dir, "apiq_init")
    )  # save adapter weights
    model.unload()
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "weight_quantizer"):
                del module.weight_quantizer
    model.base_model.save_pretrained(args.save_dir)  # save base model (fake quant)
    tokenizer.save_pretrained(args.save_dir)

    if args.convert_to_gptq:
        logging.info(f"Save base model in gptq type.")
        ## manually save config
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        quantization_config = {
            "bits": args.wbits,
            "group_size": args.group_size,
            "damp_percent": 0.01,
            "desc_act": False,
            "sym": args.symmetric,
            "true_sequential": True,
            "model_name_or_path": None,
            "model_file_base_name": "model",
            "quant_method": "gptq",
        }
        config.quantization_config = quantization_config
        config.save_pretrained(args.save_dir)

        ## save model
        def save_base_gptq_model(model):
            gptq_dicts = OrderedDict()
            for key, v in model.state_dict().items():
                if "lora" not in key:
                    new_key = key[len("base_model.model.") :]
                    new_key = new_key.replace("base_layer.", "")
                    gptq_dicts[new_key] = v.cpu()
            return gptq_dicts

        gptq_model_dicts = save_base_gptq_model(model)
        save_file(
            gptq_model_dicts,
            os.path.join(args.save_dir, "model.safetensors"),
            metadata={"format": "pt"},
        )
        tokenizer.save_pretrained(args.save_dir)  # for easy loading
        ## copy necessary files from --resume to --save_dir for easy loading
        shutil.copytree(
            os.path.join(args.resume, "apiq_init"),
            os.path.join(args.save_dir, "apiq_init"),
            dirs_exist_ok=True,
        )
        files = os.listdir(args.resume)
        for file in files:
            if file.startswith("generation") or file.endswith(".pth"):
                source_file_path = os.path.join(args.resume, file)
                destination_file_path = os.path.join(args.save_dir, file)
                shutil.copy(source_file_path, destination_file_path)
    return


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model")
    parser.add_argument("--seed", type=int, default=2)
    # Model
    parser.add_argument("--model_name_or_path", type=str)
    # parser.add_argument("--target_modules", type=str, required=True)
    parser.add_argument(
        "--peft_method", type=str, default="LoRA", choices=["LoRA", "DoRA"]
    )
    parser.add_argument(
        "--peft_args",
        type=str,
        default='{"lora_alpha": 16, "r": 64, "lora_dropout": 0}',
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        required=False,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation that the model works with",
    )
    # Calibration data
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples"
    )
    parser.add_argument(
        "--seqlen", type=int, default=1024, help="Sequence length of calibration sample"
    )
    # Quantization
    parser.add_argument(
        "--quant_method", type=str, default="default", choices=["default", "BinaryMoS", "DB-LLM"]
    )
    # For DB-LLM
    parser.add_argument(
        "--lambda_reg", type=float, default=0.0, help="Regularization parameter"
    )
    parser.add_argument(
        "--regularization_target",
        type=str,
        default="all",
        choices=["all", "quantization_params"],
        help="Parameters to apply regularization",
    )
    parser.add_argument(
        "--lwc",
        default=False,
        action="store_true",
        help="activate learnable weight clipping",
    )
    parser.add_argument(
        "--wbits", type=int, default=4, choices=[1, 2, 3, 4], help="Weight bit-width"
    )
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument(
        "--symmetric", default=False, action="store_true", help="Symmetric quantization"
    )
    parser.add_argument(
        "--disable_zero_point",
        default=False,
        action="store_true",
        help="Quantization without zero_point",
    )
    parser.add_argument(
        "--num_expert", type=int, default=8, help="Number of experts for BinaryMoS"
    )
    parser.add_argument(
        "--zero_point_type",
        type=str,
        default=None,
        choices=["output_channel", "input_channel"],
        help="Type of zero point for BinaryMoS quantization",
    )
    parser.add_argument(
        "--train_original_weights",
        default=False,
        action="store_true",
        help="Train original weights for BinaryMoS",
    )
    parser.add_argument(
        "--real_quant",
        default=False,
        action="store_true",
        help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, "
        "the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.",
    )
    # Training
    parser.add_argument(
        "--lwc_lr",
        type=float,
        default=0.005,
        help="Learning rate for weight quantization factors",
    )
    parser.add_argument(
        "--use_cosine_lr_scheduler",
        default=False,
        action="store_true",
        help="Use cosine learning rate scheduler for all trainable parameters",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
        help="Warmup ratio for cosine learning rate scheduler",
    )
    parser.add_argument(
        "--peft_lr",
        type=float,
        default=0.0005,
        help="Learning rate for PEFT parameters",
    )
    parser.add_argument(
        "--lwc_wd",
        type=float,
        default=0.1,
        help="Weight decay for weight quantization factors",
    )
    parser.add_argument(
        "--peft_wd", type=float, default=0.1, help="Weight decay for PEFT parameters"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--aug_loss",
        default=False,
        action="store_true",
        help="calculate additional loss with same quant input",
    )
    # Output
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        type=str,
        help="Cache dir of dataset, leading to faster debug",
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--save_dir", default="./models/", type=str, help="Direction for saving model"
    )
    parser.add_argument(
        "--convert_to_gptq",
        default=False,
        action="store_true",
        help="convert the base model to gptq type for real memory saving during finetuning.",
    )
    # Other
    parser.add_argument("--eval_ppl", default=False, action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Number of samples in evaluation for debug purpose.",
    )

    args = parser.parse_args()
    return args


def cli_main():
    args = arg_parse()
    logging.info(sys.argv)
    logging.info(args)
    main(args)


if __name__ == "__main__":
    cli_main()
