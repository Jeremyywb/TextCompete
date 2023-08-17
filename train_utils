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


from torch.utils.data import DataLoader
from TextCompete.data_utils.dataset import batch_to_device
from TextCompete.data_utils.dataset import CommonLitDataset,collate
from transformers import AutoTokenizer,AutoConfig

from TextCompete.basemodel.models import CommonLitModel
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
#	 from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
#	 logger = getLogger(__name__)
#	 logger.setLevel(INFO)
#	 handler1 = StreamHandler()
#	 handler1.setFormatter(Formatter("%(message)s"))
#	 handler2 = FileHandler(versionName=f"{filename}.log")
#	 handler2.setFormatter(Formatter("%(message)s"))
#	 logger.addHandler(handler1)
#	 logger.addHandler(handler2)
#	 return logger

# if args.platform['isgoogle']:
# 	args.checkpoints_path = args.platform['google']['opath']
# else:
# 	args.checkpoints_path = args.platform['featurize']['opath']
# if not os.path.exists(args.checkpoints_path):
# 	os.makedirs(args.checkpoints_path)


#=======================
# sched & optim
#=======================



def get_optim_sched(model,args):
	no_decay = ["bias", "LayerNorm.weight"]
	no_decay_para = []
	decay_para = []
	for n,p in model.named_parameters():
		if any(nd in n for nd in no_decay):
			decay_para.append(p)
		else:
			no_decay_para.append(p)
	optimizer_grouped_parameters = [
		{"params": no_decay_para,
		"weight_decay": args.optimizer["params"]['weight_decay'],
		"lr": args.optimizer["params"]['lr']},
		{"params": decay_para,
		  "weight_decay": 0.0,
		  "lr": args.optimizer["params"]['lr'],}
	  ]

	if args.optimizer['name']=="optim.AdamW":
		optimizer = eval(args.optimizer['name'])(optimizer_grouped_parameters,lr=args.optimizer["params"]['lr'])
	else:
		optimizer = eval(args.optimizer['name'])(model.parameters(), **args.optimizer['params'])

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

	return ( optimizer, scheduler, no_decay_para + decay_para )



#=======================
# step
#=======================


def training_step(args,model,criterion,inputs, prompt_inputs):
	model.train()
	device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
	inputs, prompt_inputs = collate(inputs, prompt_inputs)
	inputs, prompt_inputs = batch_to_device(inputs, prompt_inputs, device)

	if args.trainer['use_amp']:
		with amp.autocast():
			pred = model(data)
	else:
		pred = model(data)
	loss = criterion(pred, inputs['target'])
	return loss,{"train_loss":loss.item()},pred

def evaluate_step(args,model,criterion,val_loader):
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
	m,c = comp_metric(ytrue,ypred) 
	met = {"val_loss":m}   
	cols = args.model['target']
	for i,col in enumerate(cols):
		met[col] = c[i]

	return met, ytrue, ypred


#=======================
# one fold
#=======================

def fit_net(
	model,
	train_dataset,
	val_dataset,
	train_logger,
	args,
	fold,
	tokenizer,
			):
	device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
	criterion_tr = eval(args.model['loss'])(loss_name = args.model['sub_loss'],
						loss_param = args.model['sub_loss_param'],
						reduction=args.model['loss_reduction'],
						weights = args.model['target_weights']['avg'],
						device = device
						).to(device)


	train_loader = DataLoader(train_dataset,**args.train_loader)
	val_loader = DataLoader(val_dataset,**args.val_loader)
	
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

	optimizer, scheduler, optim_paras = get_optim_sched(model,args)

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
		#	 evaluation_steps = [(nb_step_per_epoch//2)+x for x in np.arange(0,nb_step_per_epoch//2,nb_step_per_epoch//(2*nstep_val))][1:]
		# else:
		#	 evaluation_steps = [x for x in np.arange(nb_step_per_epoch) if (x + 1) % step_val == 0][1:]

		trn_loss = []
		pbar = tqdm(train_loader)
		for step,(inputs, prompt_inputs) in enumerate(pbar):
			if step==epoch and step==0:
				print('\n')
				print(" ".join(train_dataset.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])))
				print('\n')
			loss,tr_sc,pred= training_step(args,model,criterion_tr,inputs, prompt_inputs)
			pbar.set_postfix(tr_sc)
			trn_loss.append(tr_sc['train_loss'])
			trn_metric["train_loss"] = np.mean(trn_loss)
 

			if args.trainer['use_amp']:
				scaler.scale(loss).backward()
				 # gradient clipping
				if args.trainer['grad_clip']:
					torch.nn.utils.clip_grad_norm_(
						 parameters = optim_paras, 
						 max_norm = args.trainer['max_norm']
					)

				scaler.step(optimizer)
				scaler.update()
				

			else:
				loss.backward()
				# gradient clipping
				if args.trainer['grad_clip']:
					torch.nn.utils.clip_grad_norm_(
						 parameters = optim_paras, 
						 max_norm=args.trainer['max_norm']
												)
				optimizer.step()

			optimizer.zero_grad()
			scheduler.step()
			del inputs, prompt_inputs, pred

		# Evaluation
		metric_val, val_label, prediction = evaluate_step(args,model,criterion_tr,val_loader)
		_score = metric_val[args.callbacks['metric_track']]
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
		msg += f'\n{nb_step_per_epoch}/{nb_step_per_epoch}  [=======================]'
		
		elapsed_time = time.time() - start_time
		epoch_time_s = int(elapsed_time)
		epoch_time_ms = int( elapsed_time/ nb_step_per_epoch*1000)
		msg += f" - {epoch_time_s}s {epoch_time_ms}ms/step - "

		for metric_name, metric_value in self._epoch_metrics.items():
			msg += f"{metric_name:<3}: {metric_value:.6f} - "
		msg += f"| {str(datetime.timedelta(seconds=elapsed_time)) + 's':<6}" 
		print(msg)
	torch.cuda.empty_cache()
	return best_predictions, val_label




			# if (((step + 1) in evaluation_steps) or (step + 1 == nb_step_per_epoch)) and (epoch>=args.callbacks["start_eval_epoch"]):
			#	 metric_val, val_label, prediction = evaluate_step(args,model,criterion_tr,val_loader)

			#	 if es:
			#		 if args.callbacks['mode']=='min':
			#			 if (metric_val[args.callbacks['metric_track']]<best):
			#				 best = metric_val[args.callbacks['metric_track']]
			#		 else:
			#			 if (metric_val>best):
			#				 best = metric_val[args.callbacks['metric_track']]

			#	 metrics = {
			#		 "fold":fold,
			#		 "epoch": epoch+1,
			#		 "step": int(step),
			#		 "global_step":step+(epoch*nb_step_per_epoch),
			#		 "best":best
					
			#	 }
			#	 metrics.update(metric_val)
			#	 metrics.update(trn_metric)

			#	 saver.log(model, metrics)
		
			#	 elapsed_time = time.time() - start_time
			#	 elapsed_time = elapsed_time * args.callbacks['verbose_eval']

			#	 lr = scheduler.get_lr()[0]
				
			#	 val_text = " "
			#	 for k,v in metric_val.items():
			#		 val_text+=f" {k}={v:.4f} "

			#	 trn_text = " "
			#	 for k,v in trn_metric.items():
			#		 trn_text+=f" {k}={v:.4f} "

			#	 metrics.update({"lr":lr})

			#	 texte = f"Epoch {epoch + 1}.{int(np.ceil((step+1)/step_val))}/{args.trainer['epochs']} lr={lr:.6f} t={elapsed_time:.0f}s "
			#	 texte = texte+trn_text+val_text
			#	 print(texte)
			#	 metric_val = metric_val[args.callbacks['metric_track']] 


		# if es:
		#	 if args.callbacks['mode']=='min':
		#		 if (best<best_epoch):
		#			 best_epoch = best
		#			 es_step = 0
		#		 else:
		#			 es_step+=1
		#			 print(f"es step {es_step}")
		#	 else:
		#		 if (best>best_epoch):
		#			 best_epoch = best
		#			 es_step = 0
		#		 else:
		#			 es_step+=1
		#			 print(f"es step {es_step}")

		#	 if (es_step>patience):
		#		 break

	# torch.cuda.empty_cache()




def train_one_fold(
	args,
	tokenizer,
	prompt_df,
	train_summary_df,
	valid_summary_df,
	train_logger,
	fold
		):	
	device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

	dset_parameters = args.data['dataset']
	dset_parameters.update(
		{"pooling_name":args.model['pooling_params']['pooling_name']}
	)
	train_dataset = CommonLitDataset(
					tokenizer,
					prompt_df,
					train_summary_df,
					**dset_parameters
				 )

	val_dataset = CommonLitDataset(
					tokenizer,
					prompt_df,
					valid_summary_df,
					**dset_parameters
				 )
	
	model_parameters = {}
	model_parameters.update( args.model['params'] )
	_update = ['CrosConvPara','CrosenEoderPara','pooling_params','spans_pooling_params','CrosAttPara']
	for _name in _update:
		model_parameters[_name] = args.model[_name]

	model = CommonLitModel(**model_parameters).to(device) 
	model.zero_grad()	

	pred_val, labels = fit_net(
		model,
		train_dataset,
		val_dataset,
		train_logger,
		args,
		fold,
		tokenizer,
	)
	return pred_val, labels


#=======================
# k fold
#=======================

def kfold(args,summary_df, prompt_df, train_logger, ver_logger):
	k = len(args.selected_folds)
	tokenizer = AutoTokenizer.from_pretrained(args.model['model_name'])
	tokenizer.save_pretrained(Path(args.checkpoints_path)/'tokenizer/')
	config = AutoConfig.from_pretrained(args.model['model_name'])
	torch.save(config, Path(args.checkpoints_path)/'config.pth')
	ypred = []
	ytrue = []
	for i in args.selected_folds:
		
		print(f"\n=============== Fold {i+1} / {k} =============== \n")
		train_df = summary_df[summary_df['fold']!=i].reset_index(drop=True)
		valid_df = summary_df[summary_df['fold']==i].reset_index(drop=True)

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
				train_df,
				valid_df,
				train_logger,
				i
			)
		ypred.append(pred_val)
		ytrue.append(labels)
	ytrue = torch.cat(ytrue,dim=0)
	ypred = torch.cat(ypred,dim=0)
	m,c = comp_metric(ytrue,ypred) 
	met = {"val_loss":m}   
	cols = args.model['target']
	for i,col in enumerate(cols):
		met[col] = c[i]

	ver_msg = "\n\n#========================"
	ver_msg+= f"# name: {args.name}"
	ver_msg+= f"# version: {args.version}"
	ver_msg+= f"# seed: {args.seed}"
	for _name, _value in args.model['params'].items():
		ver_msg+= f"# {_name}: {_value}"
	ver_msg += "#========================"
	ver_msg += "# CV"
	for _name, _value in met.items():
		ver_msg+= f"# {_name}: {_value}"
	ver_msg += "#========================\n\n"
	ver_logger.info(ver_msg)


# args.scheduler['name']
# args.optimizer["params"]['lr']
# 
# args.pooling_params['pooling_name']
# target_weights

