import re
import os
import math
import time
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path
from transformers import AdamW
import torch 
import torch.nn as nn
from torch.cuda import amp
import torch.optim as optim
import torch.nn.functional as F
import datetime

from torch.utils.data import DataLoader
from TextCompete.data_utils.dataset import batch_to_device
from TextCompete.data_utils.dataset import *
from transformers import AutoTokenizer,AutoConfig

from TextCompete.basemodel.models import *
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup,get_polynomial_decay_schedule_with_warmup
from sklearn.metrics import mean_squared_error

from tqdm.auto import tqdm
import gc
import torch.utils.checkpoint
from TextCompete.metrics_loss.loss_function import RMSELoss,get_score,CommonLitLoss



#=======================
# AverageMeter
#=======================

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
#=======================
# sched & optim
#=======================

def get_optim_sched(model,args):
    _div_warm = args.trainer['accum_iter'] if args.trainer['accum_iter']>0 else 1
    _epc_step = args.dataset_size//args.train_loader['batch_size']
    _tot_step = args.trainer['epochs'] * _epc_step//_div_warm
    _warm_ups =( _tot_step*args.scheduler['warmup'])

    _opt_wdcy = args.optimizer["params"]['weight_decay']
    _num_free = args.model['params']['freezing']
    _LR = args.optimizer["params"]['lr']
    LLDR = args.optimizer['LLDR']
    no_decay = ["bias", "LayerNorm.weight"]
    
    opt_gp_paras = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n],
         "lr": args.optimizer["HeadLR"], "weight_decay": 0.0},
        ]

    if _num_free>0:
        layers = list(model.backbone.encoder.layer[ _num_free:])
    else:
        layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()

    #===========
    #LLDR
    #===========

    for i,layer in enumerate(layers):
        _LR = _LR*LLDR**i
        opt_gp_paras += [
            { "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
              "weight_decay": _opt_wdcy, "lr": _LR},
            { "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
              "weight_decay": 0.0, "lr": _LR,
            } ]

    #=================
    # temp test
    #=================

    # opt_gp_paras = [
    #         {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
    #          'lr': args.optimizer["params"]['lr'], 'weight_decay': 0.01},
    #         {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)],
    #          'lr': args.optimizer["params"]['lr'], 'weight_decay': 0.0},
    #         {'params': [p for n, p in model.named_parameters() if "backbone" not in n],
    #          'lr': args.optimizer["params"]['lr'], 'weight_decay': 0.0}
    #     ]
    if args.optimizer['name']=="optim.AdamW":
        optimizer = eval(args.optimizer['name'])(opt_gp_paras,**args.optimizer["params"])
    elif args.optimizer['name']=="AdamW":
        print("use Debiasing Omission In BertADAM")
        optimizer = eval(args.optimizer['name'])(opt_gp_paras, **args.optimizer['tparams'])
    
    #===========
    # scheduler
    #===========

    if args.scheduler['name'] == 'poly':
        params = args.scheduler['params']
        power = params['power']
        lr_end = params['lr_end']
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, _warm_ups, _tot_step, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, _warm_ups, _tot_step)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, _warm_ups, _tot_step,num_cycles=0.5)
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:

        scheduler = eval(args.scheduler['name'])(
                 optimizer,max_lr=args.optimizer['params']['lr'],
                 epochs=args.trainer['epochs'],
                 steps_per_epoch=_epc_step,
                 pct_start = args.scheduler['warmup']
              )

    return ( optimizer, scheduler )



#=======================
# step
#=======================


# def get_score(args, _name, ytrue, ypred):
#     m,c = comp_metric(ytrue,ypred) 
#     met = {_name:m}   
#     cols = args.model['target_cols']
#     for i,col in enumerate(cols):
#         met[f'{_name[0].upper()}{col}'] = c[i]
#     return met

def training_step(args,model,criterion,inputs, prompt_inputs, device):
    model.train()
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    inputs, prompt_inputs = collate(inputs, prompt_inputs)
    inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)
    if args.trainer['use_amp']:
        with amp.autocast():
            pred = model(inputs, prompt_inputs)
            loss = criterion(pred, inputs['target'])
    else:
        pred = model(inputs, prompt_inputs)
        loss = criterion(pred, inputs['target'])
    if loss.isnan().any():
        print(loss)
    if args.trainer['accum_iter']>0:
        loss = loss/args.trainer['accum_iter']

    #********************mark******************
    # loss = torch.clamp(loss, -1e4, 1e4)
    return (
        loss,
        {"step_loss":loss.item()},
        pred.detach().cpu(),
        inputs['target'].detach().cpu(),
        )



def inference_step(args,model,inputs, prompt_inputs):
    model.eval()
    with torch.no_grad():
        inputs, prompt_inputs = collate(inputs, prompt_inputs)
        pred = model(inputs, prompt_inputs)
    return pred


#=======================
# prepare data loader
#=======================


def get_loader( args, tokenizer,summary_df,prompt_df, fold=None ):
    dset_parameters = args.data['dataset']
    dset_parameters.update(
        {"pooling_name":args.model['pooling_params']['pooling_name'],
         'multilpool' : args.model['params']['multilpool'],
         'span_pool' : args.model['params']['span_pool'],
         'add_question' : args.data['prepare']['add_question'],
        }
    )
    if fold is not None:
        print("GET TRAIN LODERR")
        train_df = summary_df[summary_df['fold']!=fold].reset_index(drop=True)
        valid_df = summary_df[summary_df['fold']==fold].reset_index(drop=True)
        train_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        train_df,
                        **dset_parameters
                     )
        args.dataset_size = len(train_dataset)
        val_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        valid_df,
                        **dset_parameters
                         )
        train_loader = DataLoader(train_dataset,**args.train_loader)
        args.len_train_loader = len(train_loader)
        val_loader = DataLoader(val_dataset,**args.val_loader)
        return train_loader,val_loader

    dset_parameters['target_cols'] = None
    test_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        summary_df,
                        **dset_parameters
                         )
    test_loader = DataLoader(test_dataset,**args.test_loader)
    return test_loader



def init_model(args, fold=None, accelerator = None):
    model_parameters = {}
    model_parameters.update( args.model['params'] )
    _update = ['CrosConvPara','CrosenEoderPara','pooling_params','spans_pooling_params','CrosAttPara']

    for _name in _update:
        model_parameters[_name] = args.model[_name]

    if not accelerator:
        model = eval(args.version)(**model_parameters)
    else:
        model_parameters['config_path'] = f'{args.modelroot}config.pth'
        with accelerator.main_process_first():
            model = eval(args.version)(**model_parameters)
            state = torch.load(f"{args.modelroot}{args.name}_fold{fold}_best.pth",
            #                            map_location=torch.device('cpu')
                    )
            model.load_state_dict(state)
    return model


#=======================
# hooks
#=======================

def forward_wrapper(name):
    
    def forward_hook(module, input, output):
        print(f"[==================={name}===================]")
        print("x is: ", input)
        print("y is: ", output)
    return forward_hook

def backward_wrapper(name):
    
    def backward_hook(module, input, output):
        print(f"[==================={name}===================]")
        print("dx is: ", input)
        print("dy is: ", output)
    return backward_hook


# A simple hook class that returns the input and output of a layer during forward/backward pass
class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        print('===============INPUT==============',self.input)
        print('===============OUTPUT==============',self.output)
    def close(self):
        self.hook.remove()

def gradhook(name):
    def hookfn(grad):
        #******************mark**************
        out = grad.clone()
        if torch.logical_or(out.isnan().any(),out.isinf().any()):
            grad = torch.clamp(grad, -2, 2)
            grad = torch.nan_to_num(grad)
            print(f"[==================={name}===================]")
            print(out)
            all_zeros = torch.all(out == 0)
            if all_zeros:
                print('weight is all zero')
    return hookfn



#=======================
# epochs
#=======================


def evaluate_epoch(args,model,criterion,val_loader, device):
    model.eval()
    ypred = []
    ytrue = []
    loss = []
    losses = AverageMeter()
    with torch.no_grad():
        for inputs, prompt_inputs in val_loader:
            inputs, prompt_inputs = collate(inputs, prompt_inputs)
            inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)
            pred = model(inputs, prompt_inputs)
            ytrue.append(inputs['target'])
            ypred.append(pred)
            loss = criterion(pred, inputs['target'])
            if args.trainer['accum_iter']>0:
                loss = loss/args.trainer['accum_iter']
            batch_size = inputs['target'].size(0)
            losses.update(loss.item(), batch_size)
    
    ytrue = torch.cat(ytrue,dim=0).detach().cpu()#.numpy() 
    ypred = torch.cat(ypred,dim=0).detach().cpu()#.numpy() 

    del inputs, prompt_inputs, pred
    met = get_score(args, 'val_loss', ytrue, ypred)
    return met,losses.avg, ytrue, ypred


def train_epoch(args,model,criterion,optimizer,scheduler,train_loader, device):
    model.train()
    if args.trainer['use_amp'] and ("cuda" in str(device)):
        scaler = amp.GradScaler(enabled=True)#apex
        print("Using Amp")
    else:
        scaler = None
    log_losses = AverageMeter()
    start_time = time.time()
    optimizer.zero_grad()
    model.zero_grad()
    # Init Metrics
    if args.trainer['accum_iter']>0:
        last_n_step = len(train_loader)%args.trainer['accum_iter']
    
    nb_step_per_epoch = args.len_train_loader
    args.steps_print = nb_step_per_epoch//3
    pbar = tqdm(train_loader)
    y_true = []
    y_pred = []
        
    #===========================
    # train loop
    #===========================

    for step,(inputs, prompt_inputs) in enumerate(pbar):
        inputs, prompt_inputs = collate(inputs, prompt_inputs)
        inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)
        if args.trainer['use_amp']:
            with amp.autocast():
                pred = model(inputs, prompt_inputs)
                loss = criterion(pred, inputs['target'])
        else:
            pred = model(inputs, prompt_inputs)
            loss = criterion(pred, inputs['target'])
        if loss.isnan().any():
            print(loss)
        if args.trainer['accum_iter']>0:
            if len(train_loader)-step<=last_n_step:
                loss = loss/last_n_step
            else:
                loss = loss/args.trainer['accum_iter']
            
            
        batch_size = inputs['target'].size(0)
        log_losses.update(loss.item(), batch_size)
        pbar.set_postfix({"step_loss":log_losses.val})#value

        if args.trainer['use_amp']:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if args.trainer['grad_clip']:
            if args.trainer['use_amp']:
                scaler.unscale_(optimizer)#useful?
            tt_norm = torch.nn.utils.clip_grad_norm_(
                 parameters = model.parameters(), 
                 max_norm = args.trainer['max_norm']
            )
        accum_iter_logic = ((step + 1) % args.trainer['accum_iter'] == 0) if args.trainer['accum_iter']>0 else True
        if  accum_iter_logic or (step + 1==len(train_loader)) :
            if args.trainer['use_amp']:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            scheduler.step()
        else:
            if args.trainer['use_amp'] and args.trainer['accum_iter']>0:
                scaler.update()
        if (
            ((step+1) % args.steps_print == 0) or
            ((step+1) == (len(train_loader)-1)) or
            tt_norm.isnan().any()
            ) :
            print('  step {0}/{1}\n    [=======================] '
              'Elapsed {remain:s} '
              'Loss: {loss.val:.4f}({loss.avg:.4f}) '
              'Grad: {grad_norm:.4f}  '
              'LR: {lr:.8f}  '
              .format(step+1, len(train_loader),
                      remain=timeSince(start_time, float(step+1)/len(train_loader)),
                      loss=log_losses,
                      grad_norm=tt_norm,
                      lr=scheduler.get_lr()[0]))

        y_true.append(inputs['target'].detach().cpu())
        y_pred.append(pred.detach().cpu())
        del inputs, prompt_inputs, pred
    return {"avg_loss":log_losses.avg}, y_true, y_pred

#=======================
# one fold
#=======================

def fit_net(
    model,
    train_loader,
    val_loader,
    train_logger,
    args,
    fold,
    tokenizer,
    device
            ):
    
    #=================
    # set loss
    #=================

    if args.model['use_weights']:
        weights = args.model['target_weights']['wtd']
    else:
        weights = args.model['target_weights']['avg']
    
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # criterion = eval(args.model['loss'])(loss_name = args.model['sub_loss'],
    #                     loss_param = args.model['sub_loss_param'],
    #                     reduction=args.model['loss_reduction'],
    #                     weights = weights,
    #                     device = device
    #                     ).to(device)
    
    criterion = nn.SmoothL1Loss(reduction='mean') .to(device)###backup
    optimizer, scheduler = get_optim_sched(model,args)
    
    mode_ = -1 if args.callbacks["mode"]=='max' else 1
    best_epoch = mode_*np.inf
    best = mode_*np.inf
    best_predictions = []
    

    #===========================
    # register hook
    #===========================
    for _name,p in model.named_parameters():
        if p.grad is not None:
            p.register_hook(gradhook(_name))

    #===========================
    # epochs
    #===========================
    
    
    for epoch in range(args.trainer['epochs']):
        # Init
        start_time = time.time()
        log_met, y_true, y_pred = train_epoch(args,model,criterion,optimizer, scheduler ,train_loader, device)
        val_met, avg_val_loss,val_label, val_pred = evaluate_epoch(args,model,criterion,val_loader, device)

        trn_met = get_score(args, 'trn_loss', torch.cat(y_true,dim=0), torch.cat(y_pred,dim=0))
        # Evaluation
        _score = val_met[args.callbacks['metric_track']]

        log_met.update(trn_met)
        log_met.update(val_met)


        #===================
        # saving best model
        #===================

        if ( mode_*_score < mode_*best ):
            log_met['status'] = f"improved from {best:.4f}!!"
            best = _score
            best_predictions = val_pred
            _name = f"{args.checkpoints_path}{args.name}_fold{fold}_{args.save_name_prefix}best.pth"
            torch.save(model.state_dict(), _name)

        #===================
        # prepare train log
        #===================

        nb_step_per_epoch = args.len_train_loader
        log_met.update({"LR":scheduler.get_lr()[0]})
        msg = f"Epoch {epoch+1}/{args.trainer['epochs']}"
        msg += f'\n{nb_step_per_epoch}/{nb_step_per_epoch}  [==============]'
        
        elapsed_time = time.time() - start_time
        epoch_time_s = int(elapsed_time)
        epoch_time_ms = int( elapsed_time/ nb_step_per_epoch*1000)
        msg += f" - {epoch_time_s}s {epoch_time_ms}ms/step - "

        for metric_name, metric_value in log_met.items():
            if metric_name == 'status':
                msg += f"{metric_name:<3}: {metric_value:<3} - "
                continue
            elif metric_name == 'LR':
                msg += f"{metric_name:<3}: {metric_value:.6f} - "
                continue
            msg += f"{metric_name:<3}: {metric_value:.4f} - "

        msg += f"| {str(datetime.timedelta(seconds=epoch_time_s)) + 's':<4}" 
        train_logger.info(msg)
    torch.cuda.empty_cache()
    return best_predictions, val_label



def train_one_fold(
    args,
    tokenizer,
    prompt_df,
    summary_df,
    train_logger,
    fold
        ):
    train_logger.info(f"#==========================FOLD{fold}==========================")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_loader(args, tokenizer, summary_df,prompt_df, fold )
    model = init_model(args).to(device)
    model.zero_grad()

    pred_val, labels = fit_net(
        model,
        train_loader,
        val_loader,
        train_logger,
        args,
        fold,
        tokenizer,
        device
    )
    return pred_val, labels




#=======================
# k fold
#=======================

def kfold(args,summary_df, prompt_df, train_logger):
    train_logger.info('\n\n----------------------START----------------------\n\n')
    tokenizer = AutoTokenizer.from_pretrained(args.model['params']['model_name'])
    tokenizer.save_pretrained(Path(args.checkpoints_path)/'tokenizer/')
    tokenizer.add_tokens(['[QUESSEP]'], special_tokens=True)
    config = AutoConfig.from_pretrained(args.model['params']['model_name'])
    torch.save(config, Path(args.checkpoints_path)/'config.pth')
    ypred = []
    ytrue = []
    for fold in args.selected_folds:
        
        print(f"\n=============== Fold {fold+1} / {len(args.selected_folds)} =============== \n")
        config = {"model":args.model}
        config.update({"optimizer":args.optimizer})
        config.update({'scheduler':args.scheduler})
        config.update({"train_loader":args.train_loader})
        config.update({"val_loader":args.val_loader})
        config.update({"trainer":args.trainer})
        config.update({"callbacks":args.callbacks})
        
        with open(args.checkpoints_path+'/params.json', 'w') as f:
            json.dump(config, f)

        pred_val, labels = train_one_fold(
                args,
                tokenizer,
                prompt_df,
                summary_df,
                train_logger,
                fold
            )
        ypred.append(pred_val)
        ytrue.append(labels)


    #======================
    # evalute msg
    #======================


    ytrue = torch.cat(ytrue,dim=0)
    ypred = torch.cat(ypred,dim=0)
    ver_log_met = get_score(args, 'val_loss', ytrue, ypred)

    ver_msg = "\n\n#========================"
    ver_msg+= f"\n# name: {args.name}"
    ver_msg+= f"\n# version: {args.version}"
    ver_msg+= f"\n# experiment: {args.data['prepare']['experiment']}"
    ver_msg+= f"\n# scheduler: {args.scheduler['name']}"
    ver_msg+= f"\n# seed: {args.seed}"
    ver_msg+= f"\n# proced_addQues: {args.data['prepare']['add_question']}"
    ver_msg+= f"\n# poll_WithQues: {args.data['dataset']['pool_question']}"

    for _name, _value in args.model['params'].items():
        ver_msg+= f"\n# {_name}: {_value}"
    ver_msg += "\n#========================"
    ver_msg += "\n# CV"
    for _name, _value in ver_log_met.items():
        ver_msg+= f"# {_name}: {_value}"
    ver_msg += "\n#========================\n\n"
    return ver_msg


def _inference(args, submission, test, prompt_df):
    accelerator = Accelerator(mixed_precision='fp16')
    grouptest = test.groupby(['prompt_id'])
    target = args.model['target_cols']
    tokenizer = AutoTokenizer.from_pretrained(args.modelroot+'/tokenizer')
    tokenizer.add_tokens(['[QUESSEP]'], special_tokens=False)
    
    for fold in args.selected_folds:
        accelerator.print(f'\n**********************\nInfering FOLD {fold}')
        model = init_model(args, fold, accelerator)
        model = accelerator.prepare(model)
        model.eval()
        for gname,gtest in grouptest:
            accelerator.print(f'FOLD {fold}\n  [============]processing {gname}...')
            test_loader = get_loader(args,tokenizer,  gtest,prompt_df)
            test_loader = accelerator.prepare(test_loader)
            ypred = []

            for inputs, prompt_inputs in test_loader:
                with torch.no_grad():
                    inputs, prompt_inputs = collate(inputs, prompt_inputs)
                    pred = model(inputs, prompt_inputs)
                pred = accelerator.gather_for_metrics(pred)
                ypred.append( pred.detach().cpu().numpy() )
            prediction = np.concatenate(ypred)
            test.loc[test['prompt_id']==gname, target] += prediction/len(args.selected_folds)
    
    #======================================================================
    #print logs and save ckpt  
    accelerator.wait_for_everyone()  
    print(test[['student_id'] + target].head())
    torch.cuda.empty_cache()
    submission = submission.drop(columns=target).merge(test[['student_id'] + target], on='student_id', how='left')
    submission[['student_id'] + target].to_csv('submission.csv', index=False)
    accelerator.print("DONE")
    #======================================================================