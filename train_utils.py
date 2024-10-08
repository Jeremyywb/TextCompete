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
from torch import autocast
import seaborn as sns
from pathlib import Path
from transformers import AdamW as transformersAdamW
from torch.optim  import AdamW as torchAdamW
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoConfig
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
import torch.utils.checkpoint


# -------------------------------------



from TextCompete.data_utils.dataset import batch_to_device, collate
from TextCompete.data_utils.dataset import get_loader
from TextCompete.metrics_loss.loss_function import CommonLitLoss,get_score,CommonLitCRMSELoss,CommonLitHuber
from torch.nn import SmoothL1Loss,GaussianNLLLoss

from TextCompete.metrics_loss.loss_function import sample as MDNPredict

from TextCompete.metrics_loss.callbacks import (
     EarlyStopping, History, get_logger
    )
from TextCompete.metrics_loss.utils import (
     IntervalStrategy, AverageMeter,AGC,calcu_grad_norm,ModelSummary,CustomAGC,get_lgb_feature
    )

from TextCompete.basemodel.models import (
    CommonLitModelV1,load_from_pretrained,make_prompt_embdeeding
    )



from TextCompete.metrics_loss.ranger21 import Ranger21

import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.mixture import GaussianMixture


def split_data_into_components(data, num_components,max_iter):
    # 使用 GMM 将数据分成 num_components 个分布
    # data : tensor
    results = []
    for component_i in range(2):
        gmm = GaussianMixture(n_components=num_components, max_iter=max_iter,random_state=0)
        gmm.fit(data[:,component_i].unsqueeze(-1))
        labels = gmm.predict(data[:,component_i].unsqueeze(-1))
        
        # 初始化结果张量
        result = torch.zeros(data[:,component_i].unsqueeze(-1).shape[0], num_components, dtype=torch.float32)
        
        # 根据每个分布，将数据分配到不同的分布中
        for i in range(num_components):
            component_data = data[:,component_i].unsqueeze(-1)[labels == i]
            component_data = component_data.squeeze().numpy()  # 将数据转换为 NumPy 数组
            result[labels == i, i] = torch.from_numpy(component_data).squeeze()
        results.append( result )
    return torch.cat(results,dim=-1)


# ==============================================================
# visual predition

def vis_realandpredict(suptitle, references, predictions):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.58))
    tencent_blue = '#0072C6'  # 腾讯蓝色
    for i, ax in enumerate(axes.flatten()):
        if i>1:
            data = predictions
        else:
            data = references
        sns.histplot(data=data[:, i % 2], ax=ax, color=tencent_blue, 
                       stat="density", common_norm=False, edgecolor='lightgray', lw=1)
        ax.set_title(f'{["References Content", "References Wording", "Predictions Content", "Predictions Wording"][i]}',
                     fontsize=12, fontstyle='italic', pad=15)
        ax.set_xlim(-2, 5)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(alpha=0.3, linestyle='--', color='lightgray')  # 添加透明网格线
        ax.spines['top'].set_color('lightgray')
        ax.spines['right'].set_color('lightgray')
        ax.spines['bottom'].set_color('lightgray')
        ax.spines['left'].set_color('lightgray')

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', fontstyle='italic', color='darkblue')
    plt.tight_layout();




def compute_loss( predict, target, var, criterions):
    n_criterions = len(criterions)
    loss = 0
    for loss_na,loss_fn in criterions.items():
        if loss_na == 'GaussianNLLLoss':
            loss += loss_fn(predict, target, var)/n_criterions
            continue
        loss += loss_fn(predict, target)/n_criterions
    return loss

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
    proced_string = f"{i+1}/{n}"
    fixlen = 1 + len(str(n))*2
    right_padding = proced_string.ljust(fixlen)
    mid = ">" if i < n-1 else '='
    proced = int(30*j)
    line = '='*(proced-1)+mid
    line = line.ljust(30,".")
    sys.stdout.write(f"{right_padding} [{line: <30}]" )
    sys.stdout.flush()

# ===========================================================

# ===========================================================
# optimizer & scheduler
def get_optim_scheduler(model,args):
    REINIT_LAYERS = args.model['params']['REINIT_LAYERS']
    num_train_steps = math.ceil(args.dataset_size/args.train_loader['batch_size'])
    total_lr_steps = (
        args.trainer['epochs'] * 
        math.ceil(num_train_steps/args.trainer['gradient_accumulation_steps'])
        )
    warmup_lr_steps =( total_lr_steps*args.scheduler['warmup'])
    OptimName = args.optimizer['name']
    schedName = args.scheduler['name']
    schedGett = args.scheduler['getter']
    schedPara = args.scheduler[schedName]
    if schedName != 'onecycle':
        schedPara.update(
            {"num_warmup_steps":warmup_lr_steps,
            "num_training_steps":total_lr_steps}
            )
    else:
        schedPara.update(
            {"epochs":args.trainer['epochs'],
            "steps_per_epoch":num_train_steps}
            )

    _LR = args.optimizer[OptimName]['lr']
    LLDR = args.optimizer['LLDR']
    lr_weight_decay = args.optimizer[OptimName]['weight_decay']
    num_freeze_layer = args.model['params']['freezing']

    no_decay =  ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # DEBUGMSG = "get_optim_scheduler"
    # DEBUGLINE= "_"*50
    
    
    # ============================================================================
    # grouped parameters
    # HEAD LAYER
    H_grouped_optimize_parameters = [
        {"ClipGroupName":"head",
         "params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": args.optimizer["HeadLR"], "weight_decay": 0.0} ]
    # ============================================================================
    
    if not args.trainer['HEADTrain']:
        if num_freeze_layer>0:
            layers = list(model.backbone.encoder.layer[ num_freeze_layer:])
        else:
            layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
        layers.reverse()

    C_grouped_optimize_parameters = []
    namefiler = lambda x:any(nd in x for nd in no_decay)
    #=======================================================
    #LLDR group parameters
    if not args.trainer['HEADTrain']:
        for i,layer in enumerate(layers):
            if REINIT_LAYERS>0 and i==0:
                layers_weight_decay = 0.0
                layers_LR = _LR
            else:
                layers_weight_decay = lr_weight_decay
                layers_LR = _LR*LLDR**i

            C_grouped_optimize_parameters += [
                { "ClipGroupName":"LLDR",
                  "params": [p for n, p in layer.named_parameters() if not namefiler(n)],
                  "weight_decay": layers_weight_decay, "lr": layers_LR},
                { "ClipGroupName":"LLDR",
                  "params": [p for n, p in layer.named_parameters() if namefiler(n)],
                  "weight_decay": 0.0, "lr": layers_LR}
                 ]
            # if i==0:
            #     DEBUGMSG  += "\n==============================================================\n"
            #     DEBUGMSG  += "    LLDR-params-not namefiler    "
            #     tmp = _LR*LLDR**i
            #     DEBUGMSG += f"\n weight_decay:{lr_weight_decay}  '_LR*LLDR**i':{tmp}   \n{DEBUGLINE}"
            #     paratmp = [n for n, p in layer.named_parameters() if not namefiler(n)]
            #     for paratmpName in paratmp:
            #         DEBUGMSG+=  f'\n{paratmpName}\n{DEBUGLINE}'
            #     DEBUGMSG  += "\n=================================\n"
            #     DEBUGMSG += "    LLDR-params-namefiler    "
            #     DEBUGMSG += f"\n weight_decay:0.0  '_LR*LLDR**i':{tmp}   \n{DEBUGLINE}"
            #     paratmp = [n for n, p in layer.named_parameters() if namefiler(n)]
            #     for paratmpName in paratmp:
            #         DEBUGMSG+=  f'\n{paratmpName}\n{DEBUGLINE}'
    #===============================================================================================
    
    if OptimName !="Ranger21":
        optimizer = eval(OptimName)(
            H_grouped_optimize_parameters+C_grouped_optimize_parameters,
            **args.optimizer[OptimName]
        )
        # DEBUGMSG+=  f"\noptimizer: {OptimName}\n{DEBUGLINE}"
        # DEBUGMSG+=  f"\nparams: {str(args.optimizer[OptimName])}\n{DEBUGLINE}"
    else:
        optimizer = Ranger21(
            H_grouped_optimize_parameters+C_grouped_optimize_parameters, 
            **args.optimizer['rparams']
        )
        # DEBUGMSG+=  f"\noptimizer：{args.optimizer['name']}\n{DEBUGLINE}"
        # DEBUGMSG+=  f"\nrparams: {str(args.optimizer['rparams'])}\n{DEBUGLINE}"


    if (not OptimName=="Ranger21"
        and args.clipgrad['clipname'] =="AGC"):
        # DEBUGMSG+=  f"\noptimizer：USE_ADAM_AGC\n{DEBUGLINE}"
        
        optimizer = AGC(
                # C_grouped_optimize_parameters, 
                # H_grouped_optimize_parameters,
                optimizer,
                **args.clipgrad['AGC']
            )

 
    #===============================================
    # poly
    scheduler = eval(schedGett)(optimizer,**schedPara)
    # DEBUGMSG += f"\nscheduler:{schedName}"
    # DEBUGMSG += "\n=============================="
    # DEBUGMSG += f"\nscheduler params:\n{str(schedPara)}\n{DEBUGLINE}"
    #============================================================
    # print(DEBUGMSG)
    return ( optimizer, scheduler )
# =====================================================================================================

    
    
# ===========================
# utils func
def _append(_this, _new):
    if _this is None: 
        _this = _new
    else:
        _this = np.append(_this, _new, axis=0)
    return _this
# ===================================================
                
class DEBUGCONTAIN:
    val = None
# ==================================================================================
# train for one folder
def train(args, model, LOGGER, criterions,device, tokenizer, trainloader, optimizer, lr_scheduler, evalloader=None):
    HISTORY = History(
        args,
        LOGGER = LOGGER,
        verbose = args.verbose
    )
    # DEBUGMSG = "======================================="
    # DEBUGLINE= "_______________________________________"
    # DEBUGCONTAIN.SINCE_last_accumulated_steps = []
    # DEBUGMSG+=  f"\ntrain"
    
    (
    num_train_steps,
    start_eval_step,
    eval_steps,
    patience) = HISTORY._prepare_args(trainloader)


    EARLY_STOPPING = EarlyStopping(patience=patience,max_minze=args.MAXIMIZE,
        verbose = args.verbose)
    HISTORY.on_train_begin(logs = {"start_time":time.time()})

    # if args.trainer['use_amp'] and ("cuda" in str(device)):
    #     scaler = amp.GradScaler(enabled=True)#apex
    optimizer.zero_grad()
    model.zero_grad()
    all_best_predictions = None

    if args.clipgrad['clipname'] == 'CustomAGC':
        CUSCLIP = CustomAGC(**args.clipgrad['CustomAGC'])


    #=========================================================
    # epoch
    for e in range(HISTORY._epochs):
        gradient_accumulation_steps = HISTORY.on_epoch_begin()
        gradient_divider = gradient_accumulation_steps
        last_gradient_divider = num_train_steps%gradient_accumulation_steps

        HISTORY._reset_to_next_eval()
        model.train()
        #=====================================================
        # each epcoh init scaler
        if args.trainer['use_amp'] and ("cuda" in str(device)):
            scaler = amp.GradScaler(enabled=True)#apex
        else:
            print("Not USEING AMP | NO GPU")
        #=====================================================
        (this_eval_targets, 
         this_eval_preditions,
         all_eval_targets, 
         all_eval_preditions,
         all_references, 
         all_predictions) = None,None,None,None,None,None

        # (this_eval_targets, this_eval_preditions, gradnorm 
        #     )= train_fn(gradient_accumulation_steps, trainloader, 
        #                     model, criterions[args.loss['losses']], optimizer, e, lr_scheduler, device)
        # =============================================
        # step
        for step, batch in enumerate(trainloader):
            # DEBUGCONTAIN.SINCE_last_accumulated_steps.append(
            #     HISTORY.SINCE_last_accumulated_steps
            # ) 

            target = batch.pop('target')
            batch = collate(batch, args)

            if args.split_n_components>1:
                n_target = split_data_into_components(target, args.split_n_components,max_iter=100000)
            else:
                n_target = target.clone()
            
            batch = batch_to_device(batch, device)
            n_target = n_target.to(device)

            # ===========================
            # transform
            
            if args.model['params']['headname'].startswith('uniform'):
                n_target = torch.round(n_target,decimals=2)
            if args.transform_target:
                n_target = n_target/2
            # ===========================

            if args.trainer['use_amp']:
                with amp.autocast(enabled=True):
                    pred, _var = model(**batch)
                    loss = compute_loss(pred, n_target, _var, criterions)
            else:
                pred, _var = model(**batch)
                loss = compute_loss(pred, n_target, _var, criterions)
            loss = loss/gradient_divider

            this_eval_targets = _append(
                this_eval_targets,
                target.numpy()
            )
            this_eval_preditions = _append(
                this_eval_preditions,
                pred.detach().cpu().numpy()
            )

            
            # print(loss)
            if args.lambda_l1>0:
                l1 = 0
                for p in model.parameters():
                    if p.grad is not None:
                          l1 = l1 + p.abs().sum()
                loss = loss + args.lambda_l1 * l1

            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # #++++++++++++++TEST++++++++++++++++++++++
            # if args.trainer['grad_clip']:
            #     grad_norm = torch.nn.utils.clip_grad_norm_(
            #              parameters = model.parameters(), 
            #              max_norm = args.trainer['max_norm']
            #     )
            # accumulation_step_msg = {
            # "PreGradNorm":grad_norm}
            # #=========================================================
            # # check for each clip step LR
            # accumulation_step_msg.update(
            #     {"accum-LR":lr_scheduler.get_lr()[0],
            #     "GradNorm":grad_norm}
            #     )
            
            # #++++++++++++++TEST++++++++++++++++++++++


            #=========================================================
            if HISTORY._verbose==1:
                status_bar(step, num_train_steps)
            HISTORY.on_step_end(loss, target)

            del target,batch, pred, n_target
            torch.cuda.empty_cache()
            gc.collect()

            if ((step+1)%gradient_accumulation_steps == 0
                or step+1 == num_train_steps):

                #====================================================================
                # for sake of the rest/last steps unmeets gradient_accumulation_steps
                if (gradient_accumulation_steps>HISTORY.UNtrained_INEpoch_steps
                    and last_gradient_divider>0):
                    gradient_divider = last_gradient_divider
                #=======================================================================

                # =======================================================
                # CLIPPING
                accumulation_step_msg = {
                "PreGradNorm":calcu_grad_norm(model.parameters())}
                if (args.clipgrad['clipname'] == 'CustomAGC'
                    and args.trainer['use_amp']):
                    scaler.unscale_(optimizer)
                    CUSCLIP.clip_grad_adptive_(model.parameters())
                elif args.clipgrad['clipname'] == 'clipnorm':
                    torch.nn.utils.clip_grad_norm_(
                             parameters = model.parameters(), 
                             max_norm = args.clipgrad['clipnorm']['max_norm']
                    )


                #=========================================================
                # check for each clip step LR
                accumulation_step_msg.update(
                    {"accum-LR":lr_scheduler.get_lr()[0],
                    "GradNorm":calcu_grad_norm(model.parameters())}
                    )
                
                # #=========================================================


                if args.trainer['use_amp']:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                model.zero_grad()
                HISTORY.on_accumulation_end(accumulation_step_msg)
                

                #==============================================================
                # evaluate on started of each epoch steps by eval_steps
                if (
                    evalloader is not None
                    and args.evaluation_strategy == IntervalStrategy.STEPS.value
                    and HISTORY.completed_steps > start_eval_step
                    and ((step+1) % eval_steps == 0 or step == num_train_steps-1)
                ):
                    
                    eval_results, all_references, all_predictions =  evaluate(args, evalloader, model, device, criterions)


                    _score = eval_results[args.callbacks['metric_track']]
                    msg = {}
                    msg.update(eval_results)

                    # ===========================================
                    # reverse
                    if args.transform_target:
                        this_eval_targets = this_eval_targets*2
                        this_eval_preditions = this_eval_preditions*2
                    # ====================================================
                    if args.split_n_components>1:
                        this_eval_preditions = torch.tensor(this_eval_preditions)
                        this_eval_preditions = this_eval_preditions.view(this_eval_preditions.shape[0], 2, -1)  # 将结果分成两部分
                        this_eval_preditions = this_eval_preditions.sum(dim=-1)
                        this_eval_preditions = this_eval_preditions.numpy()

                    all_eval_targets = _append(
                        all_eval_targets,
                        this_eval_targets
                    )
                    all_eval_preditions = _append(
                        all_eval_preditions,
                        this_eval_preditions
                    )

                    #=======================
                    # interval
                    I_train_metrics = get_score(args, 'innloss', this_eval_targets, this_eval_preditions)
                    #=====================================================================================
                    
                    msg.update(I_train_metrics)

                    #=======================
                    # global
                    G_train_metrics = get_score(args, 'glbloss', all_eval_targets, all_eval_preditions)
                    #=========================================================================
                   
                    msg.update(G_train_metrics)
                   
                    this_eval_targets, this_eval_preditions = None,None
                    
                    HISTORY.on_next_eval(step, msg)
                    EARLY_STOPPING(_score)
                    if EARLY_STOPPING._improved:
                        all_best_predictions = all_predictions
                    
                    HISTORY._reset_to_next_eval()
                    HISTORY._save_checkpoint(model, EARLY_STOPPING)

                    torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(1)
                    model.train()


                #==================================================================================

            if EARLY_STOPPING.should_training_stop:
                break
        

        if (evalloader is not None 
            and args.evaluation_strategy == IntervalStrategy.EPOCH.value
            ):
            eval_results, all_references, all_predictions =  evaluate(args, evalloader, model, device, criterions)
            _score = eval_results[args.callbacks['metric_track']]
            msg = {}
            msg.update(eval_results)

            if args.split_n_components>1:
                this_eval_preditions = torch.tensor(this_eval_preditions)
                this_eval_preditions = this_eval_preditions.view(this_eval_preditions.shape[0], 2, -1)  # 将结果分成两部分
                this_eval_preditions = this_eval_preditions.sum(dim=-1)
                this_eval_preditions = this_eval_preditions.numpy()

            all_eval_targets, all_eval_preditions = this_eval_targets, this_eval_preditions
            this_eval_targets, this_eval_preditions = None,None
            gc.collect()

            # ===========================================
            # reverse
            if args.transform_target:
                all_eval_targets = all_eval_targets*2
                all_eval_preditions = all_eval_preditions*2
            # ====================================================

            #=======================
            # global
            G_train_metrics = get_score(args, 'glbloss', all_eval_targets, all_eval_preditions)
            #=============================================================================================
            msg.update(G_train_metrics)

            # # ============tmp
            # msg.update({'Grad':gradnorm})
            step = num_train_steps-1
            
            
            HISTORY.on_next_eval(step, msg)
            EARLY_STOPPING(_score)
            if EARLY_STOPPING._improved:
                all_best_predictions = all_predictions
            
            HISTORY._reset_to_next_eval()
            HISTORY._save_checkpoint(model, EARLY_STOPPING)
            model.train()
        if HISTORY._verbose==1:
            vis_realandpredict(f'Fold {args.fold+1} Epoch {e+1} Preditct VIS',all_references,all_predictions)
        del all_predictions
        HISTORY.on_epoch_end()

        # print(DEBUGCONTAIN.SINCE_last_accumulated_steps)
            
        if (args.FullEpochStepEval
            and args.evaluation_strategy == IntervalStrategy.STEPS.value ):
            EARLY_STOPPING.should_training_stop = False
            EARLY_STOPPING._counter = 0
            # state = torch.load(args.foldModel,
            #      map_location=device
            #     )
            # model.load_state_dict(state)
            # del state
        else:
            if EARLY_STOPPING.should_training_stop:
                break
        del scaler
        torch.cuda.empty_cache()
    msg = (
        '''\n\n#========================================================================='''
        '''\n# S U M M A R Y '''
        '''\n FOLD {0}| SCORE: {1:.4f} - Completed Steps: {2}  '''
        '''\n#=========================================================================\n\n'''
    .format(args.fold+1, EARLY_STOPPING._best_score ,HISTORY.completed_steps  )
    )
    if HISTORY._verbose==1:
        print(msg)
        LOGGER.info(msg)
    
    return all_references, all_best_predictions


# ==================================================================================
# evaluate val dataloader

def evaluate(args, dataloader, model, device, criterions):
    all_predictions = None
    all_references = None
    losses = AverageMeter()

    eval_results = {}
    model.eval()
    for _, batch in enumerate(dataloader):
        # batch = collate(batch)
        # batch = batch_to_device(batch, device)
        target = batch.pop('target')
        batch = collate(batch, args)

        if args.split_n_components>1:
            n_target = split_data_into_components(target, args.split_n_components,max_iter=100000)
        else:
            n_target = target.clone()
        
        batch = batch_to_device(batch, device)
        n_target = n_target.to(device)
        # ==============================
        # transform
        if args.transform_target:
            n_target = n_target*2
        # =================================

        with torch.no_grad():
            pred, _var = model(**batch)
            loss = compute_loss(pred, n_target, _var, criterions)

        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)
        all_predictions = _append(
            all_predictions,
            pred.detach().cpu().numpy()
        )
        all_references = _append(
            all_references,
            target.numpy()
        )
        del batch, target, pred, n_target
    del loss

    if args.transform_target:
        all_references = all_references*2
        all_predictions = all_predictions*2
    if args.split_n_components>1:
        all_predictions = torch.tensor(all_predictions)
        all_predictions = all_predictions.view(all_predictions.shape[0], 2, -1)  # 将结果分成两部分
        all_predictions = all_predictions.sum(dim=-1)
        all_predictions = all_predictions.numpy()

    val_met = get_score(args, 'valloss', all_references, all_predictions)
    # eval_results.update({'VlossAvg':losses.avg})
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

def kfold(args,summary_df, prompt_df, verbose):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    #=================================
    # set weights if custom loss used
    # if args.model['use_weights']:
    #     weights = args.model['target_weights']['wtd']
    # else:
    #     weights = args.model['target_weights']['avg']
    # if args.split_n_components>1:
    #     weights = [w/args.split_n_components for w in weights]
    #     weights = [weights[0]]*args.split_n_components+[weights[1]]*args.split_n_components
    #     args.model['params']['output_dim'] = args.split_n_components*2
    # else:
    #     args.model['params']['output_dim'] = 2

    if not args.model['params']['add_prompt']:
        prompt_df = None

    #================================================

    
    import inspect
    args.fold = 0
    tokenizer, model = load_from_pretrained(args)

    if args.model['params']['add_prompt'] and args.version=='CommonLitModelV3':
        # acturally is prompts embeding dict
        summary_sentence_embs,prompt_df = make_prompt_embdeeding(summary_df,prompt_df, device, args)
        summary_sentence_embs_np = torch.stack(summary_sentence_embs,dim=0)
        del summary_sentence_embs
    # # ==========================================================
    # # model head summary
    # model = model.to(device)
    # print(model.HEAD)
    # trainloader, evalloader = get_loader( args, tokenizer,summary_df,prompt_df, 0 )
    # for batch in trainloader:
    #     break
    # forward_signature = inspect.signature(model.forward)
    # input_size = []
    # for forward_parameter in forward_signature.parameters.keys():
    #     print(forward_parameter)
    #     if forward_parameter in batch:
    #         if forward_parameter=='prompt_embedding':
    #             print(forward_parameter," with size 32,512,768")
    #             input_size.append((16,512,768))
    #         else:
    #             input_size.append((16,512))
    #             print(forward_parameter," with size 32,512")
    #     else:
    #         print(forward_parameter," with size None")
    #         input_size.append(None)
    # summary = ModelSummary(model, input_size, device)
    # summary.summary()
    # del trainloader, evalloader,forward_signature,batch
    #======================================================================== 
    del model

    oof_references, oof_preditions = None, None
    LOGGER =  get_logger(args,'train')
    lines = "#==============================================="
    msg = "\n\n==========================================>>>>>>STARTING>>>>>>==========================================\n\n"
    
    if verbose==1:
        LOGGER.info(msg)
        print(msg)
    
    reset = True
    summary_df[["content_pred","wording_pred"]] = 0
    for fold in args.selected_folds:
        msg = "\n\n{0}\nFOLD {1}/{2}\n{3}".format(lines,fold+1,len(args.selected_folds), lines )
        if verbose==1:
            LOGGER.info(msg)
            print(msg)
        args.fold = fold
        model = load_from_pretrained(args,get_tokenizer=False)
        lossNames = args.loss['losses']
        criterions = {}

        for lossNa in lossNames.split(","):
            lossPara = args.loss[lossNa]
            criterions[lossNa] =  eval(lossNa)(**lossPara).to(device) 
        if args.trainer['INITFromHead']:
            args.trainer['HEADTrain'] = False
            model.HEAD.load_state_dict(torch.load(args.headModel))


        # if reset:

        #     # ====================================================
        #     # Define max_len
        #     # ====================================================
        #     print(f"resetting max_len[{args.data['dataset']['text_max_len']}]...")
        #     lengths = []
        #     tk0 = tqdm(summary_df['text'].fillna("").values, total=len(summary_df))
        #     for text in tk0:
        #         length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        #         lengths.append(length)
        #     max_len = max(lengths) + 2 # cls & sep
        #     max_len = 640
        #     print(f"max_len: {max_len}")
        #     args.data['dataset']['text_max_len'] = max_len
        #     reset = False

        #===============================================
        # add some tok...有可能是这个原因
        # for tok in ["...","???","!!!",",,,"]:
        #     tokenizer.add_tokens([tok], special_tokens=False)
        #===============================================

        
        model = model.to(device)

        #========================================
        # register hook
        # for _name,p in model.named_parameters():
        #     if p.grad is not None:
        #         p.register_hook(gradhook(_name))
        #=======================================
        if args.version=='CommonLitModelV3':
            trainloader, evalloader = get_loader( args, tokenizer,summary_df,prompt_df,summary_sentence_embs_np, fold )
        else:
            trainloader, evalloader = get_loader( args, tokenizer,summary_df,prompt_df, fold = fold )
        
        optimizer, lr_scheduler = get_optim_scheduler(model,args)

        val_references, val_predictions = train(
            args, model, LOGGER, criterions, device, tokenizer, 
            trainloader, optimizer, lr_scheduler, evalloader)
        summary_df.loc[summary_df['fold']==fold,["content_pred","wording_pred"] ] = val_predictions
        oof_references = _append(oof_references,val_references)
        oof_preditions = _append(oof_preditions,val_predictions)
        if verbose==1:
            vis_realandpredict(f'Fold {args.fold+1} Best Preditct VIS',val_references,val_predictions)
        del  trainloader, optimizer, lr_scheduler, model, evalloader
        gc.collect()
        torch.cuda.empty_cache()
        # if args.trainer['export_lgb_feature']:
        #     get_lgb_feature(args, model, evalloader, device )

    
    # summary_df[["content_pred","wording_pred"]] = oof_preditions
    summary_df[['student_id',"content_pred","wording_pred"]].to_csv(args.platform['featurize']['opath']+"oof.csv",index=False)

    ver_log_met = get_score(args, 'oofloss', oof_references, oof_preditions)
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




