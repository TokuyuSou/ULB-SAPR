import os
from tqdm import tqdm
import torch
import torch.nn as nn
from apiq.data_utils import get_loaders

@torch.no_grad()
def evaluate(model, tokenizer, args, logging):
    logging.info("=== start evaluation ===")
    ppl_results = {}
    eval_results = {}
    if "llama" in args.model_family or "mistral" in args.model_family:
        model = model.to(args.device)
    else:
        raise ValueError("Only support llama/mistral")
    
    if args.eval_ppl:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_name_or_path.split("/")[-1]}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader)
                logging.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    tokenizer,
                    seed=args.seed,
                    seqlen=2048,
                )
                torch.save(testloader, cache_testloader)

            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // args.seqlen
            use_cache = model.config.use_cache
            model.config.use_cache = False
            model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)].to(args.device)
                # TODO: check
                if "llama" in args.model_family or "mistral" in args.model_family:
                    outputs = model.base_model.model.model(batch)
                hidden_states = outputs[0]
                logits = model.base_model.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * args.seqlen) : ((i + 1) * args.seqlen)][
                    :, 1:
                ].to(model.base_model.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * args.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * args.seqlen))
            logging.info(f'{dataset} : {ppl.item()}')
            model.config.use_cache = use_cache
            ppl_results[dataset] = ppl.item()
    
    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks
        logging.info(f"evaluating on {task_list}")
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        eval_results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        logging.info(make_table(eval_results))
        total_acc = 0
        for task in task_list:
            logging.info(f'{task} : {eval_results["results"][task]["acc,none"]*100:.2f}%')
            total_acc += eval_results['results'][task]['acc,none']
        logging.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return ppl_results, eval_results