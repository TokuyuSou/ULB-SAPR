import os
import gc
import pdb
import math
import copy
import torch
import torch.nn as nn
import numpy as np

from apiq.utils import (
    set_quant_state,
    get_lwc_parameters,
    get_peft_parameters,
    get_apiq_parameters,
    get_quantization_parameters,
    NativeScalerWithGradNormCount,
    register_scales_and_zeros,
    lwc_state_dict,
    peft_state_dict,
    get_named_linears,
    add_new_module,
    quant_inplace,
    clear_temp_variable,
    quant_temporary,
    get_learnable_parameters_from_class,
    get_all_learnable_parameters,
    calculate_regularization_term,
    get_cosine_schedule_with_warmup,
)

from apiq.model_utils import load_layer_gradients

try:
    import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
    import auto_gptq.nn_modules.qlinear.qlinear_triton as qlinear_triton
except:
    print("auto_gptq is required for real quantization")


def calibrate(model, args, dataloader, logging=None):
    logging.info("Starting ...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    is_llama = False
    if ("llama" in args.model_family) or ("mistral" in args.model_family):
        is_llama = True
        layers = model.base_model.model.model.layers
        model.base_model.model.model.embed_tokens = (
            model.base_model.model.model.embed_tokens.to(args.device)
        )
        model.base_model.model.model.norm = model.base_model.model.model.norm.to(
            args.device
        )
        num_layers = len(layers)
    else:
        raise ValueError("Only support llama/mistral now")

    layers[0] = layers[0].to(args.device)
    dtype = torch.float16
    traincast = torch.cuda.amp.autocast
    inps = torch.zeros(
        (args.nsamples, args.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=args.device,
    )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if self.is_llama:
                cache["position_ids"] = kwargs["position_ids"]
                # if "position_embeddings" in kwargs:
                #     cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    layers[0].is_llama = is_llama
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch[0].to(args.device))
            except ValueError:
                pass

    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model = model.cpu()
    if "llama" in args.model_family or "mistral" in args.model_family:
        model.base_model.model.model.embed_tokens = (
            model.base_model.model.model.embed_tokens.cpu()
        )
        model.base_model.model.model.norm = model.base_model.model.model.norm.cpu()
    else:
        raise ValueError("Only support llama/mistral now")
    torch.cuda.empty_cache()

    # same input for the first layer of fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)  # take output of fp model as input
    fp_inps_2 = (
        copy.deepcopy(inps) if args.aug_loss else None
    )  # qlayer and layer use the same quant_inps

    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.batch_size, 1, 1, 1).float()
    else:
        logging.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None

    loss_func = torch.nn.MSELoss()

    if is_llama:
        position_ids = cache["position_ids"]
        # position_embeddings = cache["position_embeddings"]
    else:
        position_ids = None
        # position_embeddings = None

    if args.resume:
        lwc_parameters = torch.load(os.path.join(args.resume, "lwc.pth"))
        peft_parameters = torch.load(os.path.join(args.resume, "peft.pth"))
    else:
        lwc_parameters = {}
        peft_parameters = {}

    for i in range(len(layers)):
        logging.info(f"=== Start quantize layer {i} ===")
        layer_gradient_dict = {}
        if args.weighted_reg:
            # Load gradients when using them for weighted regularization
            if args.gradient_dir is not None:
                layer_gradient_dict = load_layer_gradients(
                    args.gradient_dir, i, args.model_family, logging=logging, device=args.device
                )
            else:
                raise ValueError("Gradient directory is required for weighted regularization")
        layer = layers[i].to(args.device)
        qlayer = copy.deepcopy(layer)
        qlayer = qlayer.to(args.device)

        # obtain output of full-precision model
        set_quant_state(qlayer, weight_quant=False)
        if args.epochs + args.wpt_epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(
                            fp_inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            # position_embeddings=position_embeddings,
                        )[0]
                        if args.aug_loss:
                            fp_inps_2[j] = qlayer(
                                quant_inps[j].unsqueeze(0),
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                # position_embeddings=position_embeddings,
                            )[0]

        if args.resume:
            qlayer.load_state_dict(lwc_parameters[i], strict=False)
            qlayer.load_state_dict(peft_parameters[i], strict=False)

        # Save initial quantization parameters
        # (Legacy) This is used for the old regularization method
        # initial_params = {}
        # if args.regularization_target == "all":
        #     for name, param in qlayer.named_parameters():
        #         if param.requires_grad:
        #             initial_params[name] = param.detach().clone()
        # elif args.regularization_target == "quantization_params":
        #     quantization_params = list(
        #         get_quantization_parameters(qlayer, args.quant_method, name=True)
        #     )
        #     for name, param in qlayer.named_parameters():
        #         if param.requires_grad and name in quantization_params:
        #             initial_params[name] = param.detach().clone()

        # Get lambda_reg
        lambda_reg = args.lambda_reg * (args.lambda_reg_multiplier ** i)
        lambda_loss = 1.0 # coefficient for loss term

        if args.epochs + args.wpt_epochs > 0:
            ## (Note) First train using weight preservation objective for args.wpt_epochs epochs, and then train using output preservation objective for args.epochs epochs
            with torch.no_grad():
                qlayer.float()  # required for AMP training
            # create optimizer
            peft_params = {
                "params": get_peft_parameters(qlayer, args.peft_method),
                "lr": args.wpt_peft_lr,
                "weight_decay": args.wpt_peft_wd,
            }
            quantization_params = {
                "params": get_quantization_parameters(qlayer, args.quant_method),
                "lr": args.wpt_lwc_lr,
                "weight_decay": args.wpt_lwc_wd,
            }

            optimizer = torch.optim.AdamW([
                peft_params,
                quantization_params
            ])

            if args.use_cosine_lr_scheduler:
                # Make a dedicated Cosine LR scheduler for quantization parameters
                num_training_steps = args.epochs * args.nsamples // args.batch_size
                num_warmup_steps = int(args.warmup_ratio * num_training_steps)
                lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps, num_training_steps
                )

            loss_scaler = NativeScalerWithGradNormCount()

            for epoch in range(args.epochs + args.wpt_epochs):
                if epoch == args.wpt_epochs:
                    # Switch from weight preservation to output preservation
                    logging.info("Switching from weight preservation to output preservation")
                    optimizer.param_groups[0]["lr"] = args.peft_lr
                    optimizer.param_groups[0]["weight_decay"] = args.peft_wd
                    optimizer.param_groups[1]["lr"] = args.lwc_lr
                    optimizer.param_groups[1]["weight_decay"] = args.lwc_wd

                # Uncomment this if you don't use the output preservation objective while doing weight preservation training
                # Set correct coefficient for regularization term
                # if epoch < args.wpt_epochs:
                #     lambda_loss = 0
                # else:
                #     lambda_loss = 1.0
                #     lambda_reg = 0

                if epoch < args.wpt_epochs:
                    pass
                else:
                    lambda_reg = 0

                loss_list = []
                reg_loss_list = []
                norm_list = []
                for j in range(args.nsamples // args.batch_size):
                    # set_quant_state(qlayer, weight_quant=True) # No need to quantize again here.
                    if args.quant_method in ["default", "DB-LLM"]:
                        quant_temporary(qlayer)
                    loss = 0
                    if lambda_loss > 0:
                        index = j * args.batch_size
                        with traincast():
                            quant_out = qlayer(
                                quant_inps[index : index + args.batch_size,],
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids,
                                # position_embeddings=position_embeddings,
                            )[0]
                            loss = loss_func(
                                fp_inps[index : index + args.batch_size,], quant_out
                            )
                            if args.aug_loss:
                                loss += loss_func(
                                    fp_inps_2[index : index + args.batch_size,], quant_out
                                )
                            loss = lambda_loss * loss

                    # Add regularization term
                    reg_loss = 0
                    ## (Legacy) This is used for the old regularization method
                    # if lambda_reg > 0:
                    #     for name, param in qlayer.named_parameters():
                    #         if param.requires_grad and name in initial_params:
                    #             reg_loss += (param - initial_params[name]).pow(2).sum()
                    #     reg_loss = lambda_reg * reg_loss
                    if lambda_reg > 0:
                        if args.quant_method not in ["default", "DB-LLM"]:
                            raise ValueError("Regularization only supports default and DB-LLM now")
                        if args.weighted_reg:
                            # Weighted regularization term
                            reg_loss = calculate_regularization_term(
                                qlayer,
                                args.reg_method,
                                use_gradient_weighting=True,
                                gradient_dict=layer_gradient_dict,
                            )
                        else:
                            reg_loss = calculate_regularization_term(
                                qlayer, args.reg_method
                            )
                        reg_loss = lambda_reg * reg_loss

                    total_loss = loss + reg_loss

                    if not math.isfinite(total_loss.item()):
                        logging.info("Loss is NAN, stopping training")
                        # pdb.set_trace()

                    if lambda_loss > 0:
                        loss_list.append(loss.detach().cpu())
                    if lambda_reg > 0:
                        reg_loss_list.append(reg_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(
                        total_loss,
                        optimizer,
                        parameters=get_all_learnable_parameters(
                            qlayer, args.quant_method, args.peft_method
                        ),
                    ).cpu()
                    norm_list.append(norm.data)

                    # Update learning rate if using a scheduler
                    if args.use_cosine_lr_scheduler:
                        lr_scheduler.step()
                if lambda_loss > 0:
                    loss_mean = torch.stack(loss_list).mean()
                if lambda_reg > 0:
                    reg_loss_mean = torch.stack(reg_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                if lambda_reg > 0 and lambda_loss > 0:
                    logging.info(
                        f"layer {i} epoch {epoch} \t|| MSE loss: {loss_mean}\t reg loss: {reg_loss_mean}\t"
                        f"norm: {norm_mean}\tmax memory_allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2}"
                    )
                elif lambda_loss > 0:
                    logging.info(
                        f"layer {i} epoch {epoch} \t|| MSE loss: {loss_mean}\t"
                        f"norm: {norm_mean}\tmax memory_allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2}"
                    )
                elif lambda_reg > 0:
                    logging.info(
                        f"layer {i} epoch {epoch} \t|| reg loss: {reg_loss_mean}\t"
                        f"norm: {norm_mean}\tmax memory_allocated: {torch.cuda.max_memory_allocated(args.device) / 1024**2}"
                    )
                else:
                    raise ValueError("Both lambda_loss and lambda_reg are 0. Nothing is being trained.")
            clear_temp_variable(qlayer)
            del optimizer

        qlayer.half()
        if args.quant_method in ["default", "DB-LLM"]:
            quant_inplace(qlayer)

        if args.epochs + args.wpt_epochs > 0:
            # update input of quantization model
            with torch.no_grad():
                with traincast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(
                            quant_inps[j].unsqueeze(0),
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            # position_embeddings=position_embeddings,
                        )[0]
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")
            lwc_parameters[i] = lwc_state_dict(qlayer, quant_method = args.quant_method)
            peft_parameters[i] = peft_state_dict(qlayer, args.peft_method)
            torch.save(lwc_parameters, os.path.join(args.save_dir, f"lwc.pth"))
            torch.save(peft_parameters, os.path.join(args.save_dir, f"peft.pth"))
        else:
            register_scales_and_zeros(qlayer)
            layers[i] = qlayer.to("cpu")

        if args.real_quant or args.convert_to_gptq:
            assert args.wbits in [2, 3, 4], "Only support weight quantization in 2/3/4"
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0, -1)
                zeros = zeros.view(dim0, -1)
                if args.wbits == 3:
                    q_linear = qlinear_cuda.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        not module.bias is None,
                    )
                else:
                    q_linear = qlinear_triton.QuantLinear(
                        args.wbits,
                        group_size,
                        module.in_features,
                        module.out_features,
                        not module.bias is None,
                    )
                q_linear.pack(module.cpu(), scales.float().cpu(), zeros.float().cpu())
                add_new_module(name, qlayer, q_linear)
                print(f"pack quantized {name} finished")
                del module

        del layer
        del layer_gradient_dict
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    del fp_inps_2
    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache

    logging.info(model)
    return model
