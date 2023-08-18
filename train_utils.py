import re
import os
import math
import time
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path

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


from tqdm.auto import tqdm
import gc
import torch.utils.checkpoint
from TextCompete.metrics_loss.loss_function import RMSELoss,mcrmse,comp_metric,CommonLitLoss


#=======================
# model version logger
#=======================

# notebook
# def _logger(versionName=versionName):
#     from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
#     logger = getLogger(__name__)
#     logger.setLevel(INFO)
#     handler1 = StreamHandler()
#     handler1.setFormatter(Formatter("%(message)s"))
#     handler2 = FileHandler(versionName=f"{filename}.log")
#     handler2.setFormatter(Formatter("%(message)s"))
#     logger.addHandler(handler1)
#     logger.addHandler(handler2)
#     return logger

# if args.platform['isgoogle']:
#     args.checkpoints_path = args.platform['google']['opath']
# else:
#     args.checkpoints_path = args.platform['featurize']['opath']
# if not os.path.exists(args.checkpoints_path):
#     os.makedirs(args.checkpoints_path)


#=======================
# sched & optim
#=======================



def get_optim_sched(model,args):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    layerwise_learning_rate_decay = args.optimizer['LLDR']
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "weight_decay": 0.0,
            "lr": args.optimizer["HeadLR"]
        },
    ]
    if args.model['params']['freezing']>0:
        layers =  list(model.backbone.encoder.layer[args.model['params']['freezing']:])
    else:
        layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()
    lr = args.optimizer["params"]['lr']
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.optimizer["params"]['weight_decay'],
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])
    # if args.optimizer['name']=="optim.AdamW":
    #     optimizer = eval(args.optimizer['name'])(optimizer_grouped_parameters,lr=args.optimizer["params"]['lr'])
    # else:
    #     optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

    # if 'scheduler' in args:
    if args.scheduler['name'] == 'poly':

        params = args.scheduler['params']

        power = params['power']
        lr_end = params['lr_end']

        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))

        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps, training_steps, lr_end, power)

    elif args.scheduler['name'] in ['linear','cosine']:
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        if args.scheduler['name']=="linear":
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, training_steps)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps)
            
    elif args.scheduler['name'] in ['optim.lr_scheduler.OneCycleLR']:
        max_lr = args.optimizer['params']['lr']
        warmup_steps = args.scheduler['warmup'] * (args.dataset_size// (args.train_loader['batch_size']))
        training_steps = args.trainer['epochs'] * (args.dataset_size// (args.train_loader['batch_size']))
        scheduler = eval(args.scheduler['name'])(optimizer,max_lr=max_lr,
                                                 epochs=args.trainer['epochs'],
                                                 steps_per_epoch=training_steps,
                                                 pct_start = args.scheduler['warmup']
                                                 )

    return ( optimizer, scheduler )



#=======================
# step
#=======================


def get_score(args, _name, ytrue, ypred):
    m,c = comp_metric(ytrue,ypred) 
    met = {_name:m}   
    cols = args.model['target_cols']
    for i,col in enumerate(cols):
        met[f'{_name[0].upper()}{col}'] = c[i]
    return met

def training_step(args,model,criterion,inputs, prompt_inputs):
    model.train()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    inputs, prompt_inputs = collate(inputs, prompt_inputs)
    inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)

    if args.trainer['use_amp']:
        with amp.autocast():
            pred = model(inputs, prompt_inputs)
    else:
        pred = model(inputs, prompt_inputs)
    loss = criterion(pred, inputs['target'])
    return (
        loss,
        {"train_loss":loss.item()},
        pred.detach().cpu(),
        inputs['target'].detach().cpu(),
        )

def evaluate_epoch(args,model,criterion,val_loader):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    criterion = eval(args.model['loss'])(reduction="none").to(device)

    model.eval()
    ypred = []
    ytrue = []
    loss = []
    with torch.no_grad():
        for inputs, prompt_inputs in val_loader:
            inputs, prompt_inputs = collate(inputs, prompt_inputs)
            inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)
            pred = model(inputs, prompt_inputs)
            ytrue.append(inputs['target'])
            ypred.append(pred)
    
    ytrue = torch.cat(ytrue,dim=0).detach().cpu()#.numpy() 
    ypred = torch.cat(ypred,dim=0).detach().cpu()#.numpy() 

    del inputs, prompt_inputs, pred

    met = get_score(args, 'val_loss', ytrue, ypred)
    # m,c = comp_metric(ytrue,ypred) 
    # met = {"val_loss":m}   
    # cols = args.model['target_cols']
    # for i,col in enumerate(cols):
    #     met[col] = c[i]

    return met, ytrue, ypred

def inference_step(args,model,inputs, prompt_inputs):
    model.eval()
    with torch.no_grad():
        inputs, prompt_inputs = collate(inputs, prompt_inputs)
        pred = model(inputs, prompt_inputs)
    return pred


#=======================
# prepare data loader
#=======================


def get_loader( args, summary_df,prompt_df, fold=None ):


    dset_parameters = args.data['dataset']
    dset_parameters.update(
        {"pooling_name":args.model['pooling_params']['pooling_name'],
         'multilpool' : args.model['params']['multilpool'],
         'span_pool' : args.model['params']['span_pool'],
         'add_question' : args.data['prepare']['add_question'],
        }
    )
    if fold:
        train_df = summary_df[summary_df['fold']!=fold].reset_index(drop=True)
        valid_df = summary_df[summary_df['fold']==fold].reset_index(drop=True)
        train_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        train_df,
                        **dset_parameters
                     )

        val_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        valid_df,
                        **dset_parameters
                         )
        train_loader = DataLoader(train_dataset,**args.train_loader)
        val_loader = DataLoader(val_dataset,**args.val_loader)
        return train_loader,val_loader
    dset_parameters['target_cols'] = None
    test_dataset = CommonLitDataset(
                        tokenizer,
                        prompt_df,
                        valid_summary_df,
                        **dset_parameters
                         )
    test_loader = DataLoader(test_dataset,**args.val_loader)
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
        args.model['params']['config_path'] = f"{args.modelroot}{args.name}_fold{fold}_best.pth"
        with accelerator.main_process_first():
            model = eval(args.version)(**model_parameters)
    return model


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
            ):
    

    if args.model['use_weights']:
        weights = args.model['target_weights']['wtd']
    else:
        weights = args.model['target_weights']['avg']
    
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    criterion_tr = eval(args.model['loss'])(loss_name = args.model['sub_loss'],
                        loss_param = args.model['sub_loss_param'],
                        reduction=args.model['loss_reduction'],
                        weights = weights,
                        device = device
                        ).to(device)
    
    args.len_train_loader = len(train_loader)
    args.dataset_size = len(train_dataset)

    mode_ = -1 if args.callbacks["mode"]=='max' else 1
    best_epoch = mode_*np.inf
    best = mode_*np.inf

    es = args.callbacks['es']
    es_step = 0
    patience = args.callbacks['patience']
  
    if args.trainer['use_amp'] and ("cuda" in str(device)):
        scaler = amp.GradScaler()
        print("Using Amp")
    else:
        scaler = None

    optimizer, scheduler = get_optim_sched(model,args)

    best_predictions = []

    for epoch in range(args.trainer['epochs']):
        # Init
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        # Init Metrics
        trn_metric = {}
        for k in ["train_loss"]:
            trn_metric[k]=0
        

        nb_step_per_epoch = args.len_train_loader
        step_val = int(np.round(nb_step_per_epoch*args.callbacks['epoch_pct_eval']))
        nstep_val = int(1/args.callbacks['epoch_pct_eval'])
        # if args.callbacks['epoch_eval_dist']=="uniforme":
        #     evaluation_steps = [(nb_step_per_epoch//2)+x for x in np.arange(0,nb_step_per_epoch//2,nb_step_per_epoch//(2*nstep_val))][1:]
        # else:
        #     evaluation_steps = [x for x in np.arange(nb_step_per_epoch) if (x + 1) % step_val == 0][1:]

        trn_loss = []
        pbar = tqdm(train_loader)
        y_true = []
        y_pred = []
        for step,(inputs, prompt_inputs) in enumerate(pbar):
            # if step==epoch and step==0:
            #     print('\n')
            #     print(" ".join(train_dataset.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])))
            #     print('\n')
            loss,tr_sc,pred,_true = training_step(args,model,criterion_tr,inputs, prompt_inputs)
            pbar.set_postfix(tr_sc)
            trn_loss.append(tr_sc['train_loss'])
            trn_metric["train_loss"] = np.mean(trn_loss)
 

            if args.trainer['use_amp']:
                scaler.scale(loss).backward()
                 # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                         parameters = [p  for p in list(model.parameters()) if p.requires_grad], 
                         max_norm = args.trainer['max_norm']
                    )

                scaler.step(optimizer)
                scaler.update()
                

            else:
                loss.backward()
                # gradient clipping
                if args.trainer['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                         parameters = [p  for p in list(model.parameters()) if p.requires_grad], 
                         max_norm=args.trainer['max_norm']
                                                )
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

            y_true.append(_true)
            y_pred.append(pred)
            del inputs, prompt_inputs, pred, _true

        _trn_met = get_score(args, 'eptrain_loss', torch.cat(y_true,dim=0), torch.cat(y_pred,dim=0))
        # Evaluation
        metric_val, val_label, prediction = evaluate_epoch(args,model,criterion_tr,val_loader)
        _score = metric_val[args.callbacks['metric_track']]

        trn_metric.update(_trn_met)
        trn_metric.update(metric_val)


        #===================
        # saving best model
        #===================

        if ( mode_*_score < mode_*best ):
            trn_metric['status'] = f"improved from {best:.4f}!!"
            best = _score
            best_predictions = prediction
            _name = f"{args.checkpoints_path}{args.name}_fold{fold}_best.pth"
            torch.save(model.state_dict(), _name)

        #===================
        # prepare train log
        #===================


        trn_metric.update({"LR":scheduler.get_lr()[0]})
        msg = f"Epoch {epoch+1}/{args.trainer['epochs']}"
        msg += f'\n{nb_step_per_epoch}/{nb_step_per_epoch}  [==============]'
        
        elapsed_time = time.time() - start_time
        epoch_time_s = int(elapsed_time)
        epoch_time_ms = int( elapsed_time/ nb_step_per_epoch*1000)
        msg += f" - {epoch_time_s}s {epoch_time_ms}ms/step - "

        for metric_name, metric_value in trn_metric.items():
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




            # if (((step + 1) in evaluation_steps) or (step + 1 == nb_step_per_epoch)) and (epoch>=args.callbacks["start_eval_epoch"]):
            #     metric_val, val_label, prediction = evaluate_step(args,model,criterion_tr,val_loader)

            #     if es:
            #         if args.callbacks['mode']=='min':
            #             if (metric_val[args.callbacks['metric_track']]<best):
            #                 best = metric_val[args.callbacks['metric_track']]
            #         else:
            #             if (metric_val>best):
            #                 best = metric_val[args.callbacks['metric_track']]

            #     metrics = {
            #         "fold":fold,
            #         "epoch": epoch+1,
            #         "step": int(step),
            #         "global_step":step+(epoch*nb_step_per_epoch),
            #         "best":best
                    
            #     }
            #     metrics.update(metric_val)
            #     metrics.update(trn_metric)

            #     saver.log(model, metrics)
        
            #     elapsed_time = time.time() - start_time
            #     elapsed_time = elapsed_time * args.callbacks['verbose_eval']

            #     lr = scheduler.get_lr()[0]
                
            #     val_text = " "
            #     for k,v in metric_val.items():
            #         val_text+=f" {k}={v:.4f} "

            #     trn_text = " "
            #     for k,v in trn_metric.items():
            #         trn_text+=f" {k}={v:.4f} "

            #     metrics.update({"lr":lr})

            #     texte = f"Epoch {epoch + 1}.{int(np.ceil((step+1)/step_val))}/{args.trainer['epochs']} lr={lr:.6f} t={elapsed_time:.0f}s "
            #     texte = texte+trn_text+val_text
            #     print(texte)
            #     metric_val = metric_val[args.callbacks['metric_track']] 


        # if es:
        #     if args.callbacks['mode']=='min':
        #         if (best<best_epoch):
        #             best_epoch = best
        #             es_step = 0
        #         else:
        #             es_step+=1
        #             print(f"es step {es_step}")
        #     else:
        #         if (best>best_epoch):
        #             best_epoch = best
        #             es_step = 0
        #         else:
        #             es_step+=1
        #             print(f"es step {es_step}")

        #     if (es_step>patience):
        #         break

    # torch.cuda.empty_cache()




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

    train_loader, val_loader = get_loader( args, summary_df,prompt_df, fold )
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
    )
    return pred_val, labels


#=======================
# k fold
#=======================

def kfold(args,summary_df, prompt_df, train_logger):
    train_logger.info('\n\n----------------------START----------------------\n\n')
    k = len(args.selected_folds)
    tokenizer = AutoTokenizer.from_pretrained(args.model['params']['model_name'])
    tokenizer.save_pretrained(Path(args.checkpoints_path)/'tokenizer/')
    tokenizer.add_tokens(['[QUESSEP]'], special_tokens=True)
    config = AutoConfig.from_pretrained(args.model['params']['model_name'])
    torch.save(config, Path(args.checkpoints_path)/'config.pth')
    ypred = []
    ytrue = []
    for fold in args.selected_folds:
        
        print(f"\n=============== Fold {fold+1} / {k} =============== \n")

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
    ytrue = torch.cat(ytrue,dim=0)
    ypred = torch.cat(ypred,dim=0)
    m,c = comp_metric(ytrue,ypred) 
    met = {"val_loss":m}   
    cols = args.model['target_cols']
    for i,col in enumerate(cols):
        met[col] = c[i]

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
    for _name, _value in met.items():
        ver_msg+= f"# {_name}: {_value}"
    ver_msg += "\n#========================\n\n"
    return ver_msg



def _inference(args, submission, test, prompt_df):
    accelerator = Accelerator(mixed_precision='fp16')
    grouptest = test.groupby(['prompt_id'])
    target = args.model['target_cols']

	for fold in args.selected_folds:
		accelerator.print(f'\n**********************\nInfering FOLD {fold}')
    	model = init_model(args, fold, accelerator)
    	model = accelerator.prepare(model)
    	for gname,gtest in grouptest:
    		accelerator.print(f'FOLD {fold}\n  [============]processing {gname}...')
    		test_loader = get_loader( args, gtest,prompt_df)
    		ypred = []
    
    		for inputs, prompt_inputs in test_loader:
        		pred = inference_step(args,model,inputs, prompt_inputs)
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
