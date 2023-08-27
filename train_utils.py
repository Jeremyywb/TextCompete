import pandas as pd
import numpy as np
import random
import sys
import datetime
import time
import copy
import math
import re
import os
import gc
import json

from pathlib import Path
from transformers import AdamW
import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoConfig
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import torch.utils.checkpoint


from TextCompete.data_utils.dataset import batch_to_device, collate
from TextCompete.data_utils.dataset import get_loader
from TextCompete.metrics_loss.loss_function import RMSELoss,get_score,CommonLitLoss
from TextCompete.metrics_loss.callbacks import (
     EarlyStopping, History, get_logger
    )
from TextCompete.metrics_loss.utils import (
     IntervalStrategy, AverageMeter
    )
from TextCompete.basemodel.models import (
     load_from_pretrained, CommonLitModelV1
    )



# ==================
# new configs
# UseCusLoss: false
# gradient_accumulation_steps*train_batch = 32
# trainer:
#     gradient_accumulation_steps
# use_accumulation: true

# optimizer: LR

# beside calls
# start_eval_step: 0.5
# eval_steps: 0.2
# ==divideable gradient_accumulation_steps
# evaluation_strategy:no|steps|epoch
# es_strategy: half|one_third|a_quarter|one_fifth|epochs
# do_inference: false
# do_train: true
# ==============================================================


# ===================
# bar
def status_bar(i,n):
    sys.stdout.write('\r')
    j = (i + 1) / n
    sys.stdout.write("[%-20s] %d%%" % ('='*int(20*j), 100*j))
    sys.stdout.flush()
# ===========================================================


# ===========================================================
# optimizer & scheduler
def get_optim_scheduler(model,args):
    
    num_train_steps = math.ceil(args.dataset_size/args.train_loader['batch_size'])
    total_lr_steps = (
        args.trainer['epochs'] * 
        math.ceil(num_train_steps/args.trainer['gradient_accumulation_steps'])
        )
    warmup_lr_steps =( total_lr_steps*args.scheduler['warmup'])

    lr_weight_decay = args.optimizer["params"]['weight_decay']
    num_freeze_layer = args.model['params']['freezing']
    _LR = args.optimizer["LR"]
    LLDR = args.optimizer['LLDR']
    no_decay = ["bias", "LayerNorm.weight"]
    
    # ============================================================================
    # grouped parameters
    # HEAD LAYER
    grouped_optimize_parameters = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": args.optimizer["HeadLR"], "weight_decay": 0.0} ]
    # ============================================================================

    if num_freeze_layer>0:
        layers = list(model.backbone.encoder.layer[ num_freeze_layer:])
    else:
        layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()

    namefiler = lambda x:any(nd in x for nd in no_decay)
    #=======================================================
    #LLDR group parameters
    for i,layer in enumerate(layers):
        grouped_optimize_parameters += [
            { "params": [p for n, p in layer.named_parameters() if not namefiler(n)],
              "weight_decay": lr_weight_decay, "lr": _LR*LLDR**i},
            { "params": [p for n, p in layer.named_parameters() if namefiler(n)],
              "weight_decay": 0.0, "lr": _LR*LLDR**i}
             ]
    #===============================================================================================
    
    if args.optimizer['name']=="optim.AdamW":
        optimizer = eval(args.optimizer['name'])(
            grouped_optimize_parameters,**args.optimizer["params"]
        )
    elif args.optimizer['name']=="AdamW":
        print("use Debiasing Omission In BertADAM")
        optimizer = eval(args.optimizer['name'])(
            grouped_optimize_parameters, **args.optimizer['tparams']
        )
    
    #===============================================
    # poly
    if args.scheduler['name'] == 'poly':
        params = args.scheduler['params']
        power = params['power']
        lr_end = params['lr_end']
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, warmup_lr_steps, total_lr_steps, lr_end, power)

    #===============================================
    # linear or cosine
    elif args.scheduler['name'] in ['linear','cosine']:
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, warmup_lr_steps, total_lr_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, warmup_lr_steps, total_lr_steps,num_cycles=0.5)
    #===============================================
    # OneCycleLR
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:
        scheduler = eval(args.scheduler['name'])(
                 optimizer,max_lr=args.optimizer['params']['lr'],
                 epochs=args.trainer['epochs'],
                 steps_per_epoch= total_lr_steps,
                 pct_start = args.scheduler['warmup']
              )
    #============================================================

    return ( optimizer, scheduler )
# =====================================================================================================



# ==================
# new configs
# gradient_accumulation_steps*train_batch = 32
# gradient_accumulation_steps
# use_accumulation: true
# start_eval_step: 0.5
# eval_steps: 0.2
# optimizer: LR
# ==divideable gradient_accumulation_steps
# ==============================================================


# ===========================
# utils func
def _append(_this, _new):
    if _this is None: 
        _this = _new
    else:
        _this = np.append(_this, _new, axis=0)
    return _this
# ===================================================
                

# ==================================================================================
# train for one folder
def train(args, model, LOGGER, criterion,device, tokenizer, trainloader, optimizer, lr_scheduler, evalloader=None):
    HISTORY = History(
        args,
        LOGGER = LOGGER,
        verbose = args.verbose
    )

    (
    num_train_steps,
    start_eval_step,
    eval_steps,
    patience) = HISTORY._prepare_args(trainloader)
    EARLY_STOPPING = EarlyStopping(patience=patience,max_minze=args.MAXIMIZE) 
    HISTORY.on_train_begin(logs = {"start_time":time.time()})

    if args.trainer['use_amp'] and ("cuda" in str(device)):
        scaler = amp.GradScaler(enabled=True)#apex
    optimizer.zero_grad()
    model.zero_grad()
    all_best_predictions = None
    for _ in range(HISTORY._epochs):
        gradient_accumulation_steps = HISTORY.on_epoch_begin()
        HISTORY._reset_to_next_eval()
        model.train()
        (this_eval_targets, 
         this_eval_preditions,
         all_targets, 
         all_preditions,
         all_references, 
         all_predictions) = None,None,None,None,None,None
        for step, batch in enumerate(trainloader):
            batch = collate(batch)
            batch = batch_to_device(batch, device)
            target = batch.pop('target')

            if args.trainer['use_amp']:
                with amp.autocast():
                    pred = model(**batch)
                    loss = criterion(pred, target)
            else:
                pred = model(**batch)
                loss = criterion(pred, target)
            this_eval_targets = _append(
                this_eval_targets,
                target.detach().cpu().numpy()
            )
            this_eval_preditions = _append(
                this_eval_preditions,
                pred.detach().cpu().numpy()
            )

            loss = loss/gradient_accumulation_steps
            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            del batch, pred

            status_bar(step, num_train_steps)
            HISTORY.on_step_end(loss, target)
            del target

            if HISTORY.SINCE_last_accumulated_steps == gradient_accumulation_steps:

                #====================================================================
                # for sake of the rest/last steps unmeets gradient_accumulation_steps
                gradient_accumulation_steps = min(gradient_accumulation_steps,HISTORY.UNtrained_INEpoch_steps)
                #=======================================================================


                #=========================================================
                # check for each clip step LR
                accumulation_step_msg = {"accum-LR":lr_scheduler.get_lr()[0]}
                #=========================================================

                if args.trainer['use_amp']:
                    scaler.unscale_(optimizer)
                if args.trainer['grad_clip']:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                             parameters = model.parameters(), 
                             max_norm = args.trainer['max_norm']
                    )
                    accumulation_step_msg.update(
                            {"GradNorm":total_grad_norm}
                        )
                if args.trainer['use_amp']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
                lr_scheduler.step()
                HISTORY.on_accumulation_end(accumulation_step_msg)

                #==============================================================
                # evaluate on started of each epoch steps by eval_steps
                if (
                    evalloader is not None
                    and args.evaluation_strategy == IntervalStrategy.STEPS.value
                    and HISTORY.completed_steps > start_eval_step
                    and ((step+1) % eval_steps == 0 or step == num_train_steps-1)
                ):
                    torch.cuda.empty_cache()
                    eval_results, all_references, all_predictions =  evaluate(args, evalloader, model, device, criterion)

                    _score = eval_results[args.callbacks['metric_track']]
                    msg = {}
                    msg.update(eval_results)
                    
                    all_targets = _append(
                        all_targets,
                        this_eval_targets
                    )
                    all_preditions = _append(
                        all_preditions,
                        this_eval_preditions
                    )

                    #=======================
                    # interval
                    I_train_metrics = get_score(args, 'inter_train_loss', this_eval_targets, this_eval_preditions)
                    #=====================================================================================
                    msg.update(I_train_metrics)
                    #=======================
                    # global
                    G_train_metrics = get_score(args, 'global_train_loss', all_targets, all_preditions)
                    #=========================================================================
                    msg.update(G_train_metrics)
                    this_eval_targets, this_eval_preditions = None,None

                    
                    HISTORY.on_next_eval(step, msg)
                    EARLY_STOPPING(_score)
                    if EARLY_STOPPING._improved:
                        all_best_predictions = all_predictions
                    del all_predictions
                    HISTORY._reset_to_next_eval()
                    HISTORY._save_checkpoint(model, EARLY_STOPPING)

                #==================================================================================

            if EARLY_STOPPING.should_training_stop:
                break
        HISTORY.on_epoch_end()

        if evalloader is not None and args.evaluation_strategy == IntervalStrategy.EPOCH.value:
            eval_results, all_references, all_predictions =  evaluate(args, evalloader, model, device, criterion)
            _score = eval_results[args.callbacks['metric_track']]
            msg = {}
            msg.update(eval_results)
            
            all_targets, all_preditions = this_eval_targets, this_eval_preditions
            this_eval_targets, this_eval_preditions = None,None
            gc.collect()
            #=======================
            # global
            G_train_metrics = get_score(args, 'global_train_loss', all_targets, all_preditions)
            #=============================================================================================
            msg.update(G_train_metrics)
            
            
            HISTORY.on_next_eval(step, msg)
            EARLY_STOPPING(_score)
            if EARLY_STOPPING._improved:
                all_best_predictions = all_predictions
            del all_predictions
            HISTORY._reset_to_next_eval()
            HISTORY._save_checkpoint(model, EARLY_STOPPING)
            

        if EARLY_STOPPING.should_training_stop:
            break
    msg = (
        '''\n\n#========================================================================='''
        '''\n# S U M M A R Y '''
        '''\n FOLD {0}| SCORE: {1:.4f} - Completed Steps: {2}  '''
        '''\n#=========================================================================\n\n'''
    .format(args.fold, EARLY_STOPPING._best_score ,HISTORY.completed_steps  )
    )
    print(msg)
    LOGGER.info(msg)
    
    return all_references, all_best_predictions

# ==================================================================================
# evaluate val dataloader

def evaluate(args, dataloader, model, device, criterion):
    all_predictions = None
    all_references = None
    losses = AverageMeter()

    eval_results = {}
    model.eval()
    for _, batch in enumerate(dataloader):
        batch = collate(batch)
        batch = batch_to_device(batch, device)
        target = batch.pop('target')
        with torch.no_grad():
            pred = model(**batch)
        loss = criterion(pred, target)
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)
        all_predictions = _append(
            all_predictions,
            pred.detach().cpu().numpy()
        )
        all_references = _append(
            all_references,
            target.detach().cpu().numpy()
        )
        del batch, target, pred
    val_met = get_score(args, 'val_loss', all_references, all_predictions)
    eval_results.update({'VlossAvg':losses.avg})
    eval_results.update(val_met)
    torch.cuda.empty_cache()
    return eval_results, all_references, all_predictions


# =====================================================
# maybe outliers calls nan gradients
# this is a trial setting grad small when nan happen
# let outliers give small comtributes not none

def gradhook(name):
    def hookfn(grad):
        #******************mark**************
        out = grad.clone()
        if torch.logical_or(out.isnan().any(),out.isinf().any()):
            print(f"[==================={name}===================]")
            print(out)
            grad = torch.clamp(grad, -0.5, 0.5)
            grad = torch.nan_to_num(grad)
            all_zeros = torch.all(out == 0)
            if all_zeros:
                print('weight is all zero')
    return hookfn
# ======================================================================

def kfold(args,summary_df, prompt_df):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    

    #=================================
    # set weights if custom loss used
    if args.model['use_weights']:
        weights = args.model['target_weights']['wtd']
    else:
        weights = args.model['target_weights']['avg']
    #================================================


    if hasattr(args,'UseCusLoss'):
        if args.UseCusLoss:
            criterion = eval(args.model['loss'])(loss_name = args.model['sub_loss'],
                            loss_param = args.model['sub_loss_param'],
                            reduction=args.model['loss_reduction'],
                            weights = weights,
                            device = device
                            ).to(device)
        else:
            criterion = nn.SmoothL1Loss(reduction='mean') .to(device)
    else:
        raise ValueError('args UseCusLoss attr not seted')

    oof_references, oof_preditions = None, None
    LOGGER =  get_logger(args,'train')
    lines = "#==============================================="
    msg = "\n\n==========================================>>>>>>STARTING>>>>>>==========================================\n\n"
    LOGGER.info(msg)
    print(msg)
    for fold in args.selected_folds:
        msg = "\n\n{0}\nFOLD {1}/{2}\n{3}".format(lines,fold+1,len(args.selected_folds), lines )
        LOGGER.info(msg)
        print(msg)
        args.fold = fold
        tokenizer, model = load_from_pretrained(args)
        model = model.to(device)

        #========================================
        # register hook
        for _name,p in model.named_parameters():
            if p.grad is not None:
                p.register_hook(gradhook(_name))
        #=======================================

        trainloader, evalloader = get_loader( args, tokenizer,summary_df,prompt_df, fold )
        optimizer, lr_scheduler = get_optim_scheduler(model,args)

        val_references, val_predictions = train(
            args, model, LOGGER, criterion, device, tokenizer, 
            trainloader, optimizer, lr_scheduler, evalloader)
        
        oof_references = _append(oof_references,val_references)
        oof_preditions = _append(oof_preditions,val_predictions)
    


    ver_log_met = get_score(args, 'oof_loss', oof_references, oof_preditions)
    ver_msg = (
          "\n\n\n#========================"
          f"\n# name: {args.name}"
          f"\n# version: {args.version}"
          f"\n# experiment: {args.data['prepare']['experiment']}"
          f"\n# scheduler: {args.scheduler['name']}"
          f"\n# seed: {args.seed}"
          f"\n# proced_addQues: {args.data['prepare']['add_question']}"
          f"\n# poll_WithQues: {args.data['dataset']['pool_question']}"
          "\n#============================================"
        ) 

    for _name, _value in args.model['params'].items():
        ver_msg+= f"\n# {_name}: {_value}"
    ver_msg += "\n\n#========================"
    ver_msg += "\n# CV"
    for _name, _value in ver_log_met.items():
        ver_msg+= f"# {_name}: {_value}"
    ver_msg += "\n#============================================\n\n"
    return ver_msg,ver_log_met



