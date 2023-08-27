# =============================
# dateset
# =============================

import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

class CommonLitDataset(Dataset):
	def __init__(
		self,
		tokenizer,
		prompt_df,
		summary_df,
		pooling_name,
		multilpool,
		span_pool,
		pool_question,
		add_question,
		target_cols,
		text_max_len = 512,
		prompt_text_max_len = 512,
						 ):

		#=====================
		# add in tokenizer init
		#tokenizer.add_tokens(['[QUESSEP]'], special_tokens=True)
		#=====================

		self.pool_question = pool_question
		self.add_question = add_question
		self.pooling_name = pooling_name
		self.multilpool = multilpool
		self.span_pool = span_pool
		self.sentSEPs = ['!','?','.','\n','[CLS]','[SEP]']#CLS:1,SEP:2
		self.sentSEPs = tokenizer.convert_tokens_to_ids(self.sentSEPs)#list
		print(f"#===============================\n#{self.sentSEPs}\n#===============================")
		self.tokenizer = tokenizer
		self.text_max_len = text_max_len
		self.prompt_text_max_len = prompt_text_max_len

		self.texts = summary_df['text'].values
		self.prompt_id = summary_df['prompt_id'].values
		self.prompt_df = prompt_df

		if target_cols is not None:
			self.labels = summary_df[target_cols].values
		else:
			self.labels = None

		self._create_prompt_dict()
		self._create_samples()

	def _create_prompt_dict(self):
		if self.prompt_df is None:
			self.prompt_dict = None
		else:
			self.prompt_dict = {}

			list_prompt_inputs = [self.prepare_input(
								'prompt',
								prompt_text, 
								self.prompt_text_max_len
							)  for prompt_text in self.prompt_df['prompt_text'].values ]

			max_prompt_len = max( [ sum(inp['prompt_attention_mask']) for inp in list_prompt_inputs])
			print(f"MAX PROMPT LEN <{max_prompt_len}>")
			for prompt_id, prompt_input in zip(self.prompt_df['prompt_id'].values, list_prompt_inputs):
				for k, v in prompt_input.items():
					prompt_input[k] = prompt_input[k][:max_prompt_len]
				self.prompt_dict[prompt_id] = prompt_input

	def prepare_input(self,name_prefix,text, max_len):
		inputs = self.tokenizer.encode_plus(
						text,
						return_tensors=None,
						add_special_tokens=True,
						max_length = max_len,
						pad_to_max_length = True,#IF not sort with len,will be True
						truncation = True
				)
		for k in list(inputs):
			inputs[f'{name_prefix}_{k}'] = torch.tensor(inputs.pop(k), dtype=torch.long)
		return inputs

	def __len__(self):
		return len(self.texts)

	def __getitem__(self, _id ):
		return self.samples[_id]

	def _create_samples(self):
		self.samples = []
		for _id in range(len(self.texts)):
			features = []
			inputs = self.prepare_input('summary',self.texts[_id], self.text_max_len)
			features.extend(['summary_input_ids','summary_attention_mask'])
			if self.prompt_dict is not None:
				prompt_inputs = self.prompt_dict[self.prompt_id[_id]]
			else:
				prompt_inputs = None
			inputs_ids = inputs['summary_input_ids'].numpy()
			if self.pool_question:
				sepmask = np.isin(inputs_ids,[1,2])
				mask = sepmask.cumsum()
				mask[mask==mask[-1]]=0
				mask= (mask>0)*(~sepmask)*1
			else:
				sepmask = np.isin(inputs_ids,[1,2])
				mask = sepmask.cumsum()
				mask[mask==mask[-1]]=0
				if self.add_question:
					mask = (mask>1)*1
				else:
					mask = (mask>0)*1
				mask = mask*(~sepmask)
			if (self.multilpool or self.span_pool):
				sepmask = np.isin(inputs_ids,self.sentSEPs)
				slable = sepmask.cumsum()
				slable[slable==slable[-1]]=0
				slable[0] = 0
					
			#============================
			# mask : head atteion pooling
			# pool_mask : pooling
			#====================

			if (self.multilpool or self.span_pool) :
				inputs['summary_slable'] = torch.LongTensor(slable)
				features.append('summary_slable')

			if self.pooling_name in ['MeanPooling','MaxPooling','MinPooling']:
				inputs['summary_smask'] = torch.LongTensor(mask)
				features.append('summary_smask')
			elif self.pooling_name=="GeMText":
				inputs['summary_smask'] = inputs["summary_attention_mask"]
				features.append('summary_smask')
			if self.labels is not None:
				inputs['target'] = torch.tensor(self.labels[_id], dtype=torch.float)
				features.append('target')

			if prompt_inputs is not None:
				inputs.update(
					{'prompt_input_ids':prompt_inputs['prompt_input_ids'],
					'prompt_attention_mask':prompt_inputs['prompt_attention_mask']}
					)
				features.extend(['prompt_input_ids','prompt_attention_mask'])
			self.samples.append({k:inputs[k] for k in features})


def collate(inputs):#, sentmask):
	for intype in ['summary','prompt']:
		attention_mask = f"{intype}_attention_mask"
		if attention_mask in inputs:
			_mask_len = int(inputs[attention_mask].sum(axis=1).max())
			for k in list(inputs):
				if intype in k:
					inputs[k] = inputs[k][:, :_mask_len]
	return inputs

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

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