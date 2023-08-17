import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CommonLitDataset(Dataset):
	def __init__(
		self,
		tokenizer,
		prompt_df,
		summary_df,
		pooling_name,
		add_question,
		target_cols,
		text_max_len = 512,
		prompt_text_max_len = 512,
						 ):

		#=====================
		# add in tokenizer init
		#tokenizer.add_tokens(['[QUESSEP]'], special_tokens=True)
		#=====================

		self.add_question = add_question
		self.pooling_name = pooling_name
		self.sentSEPs = [',','.','!','?','[CLS]', '[QUESSEP]']#CLS:1
		self.sentSEPs = tokenizer.convert_tokens_to_ids(self.sentSEPs)#list
		self.quesSEP = tokenizer.convert_tokens_to_ids(['[QUESSEP]'])[0]
		self.tokenizer = tokenizer

		self.text_max_len = text_max_len
		self.prompt_text_max_len = prompt_text_max_len

		self.texts = summary_df['text'].values
		self.prompt_id = summary_df['prompt_id'].values
		if target_cols is not None:
			self.labels = summary_df[target_cols].values
		else:
			self.labels = None

		self.prompt_dict = {k:v
				for k,v in zip( 
					prompt_df['prompt_id'].values,
					self._pre_prompt_dict(prompt_df['prompt_text'].values)
					)
			}

	def _pre_prompt_dict(self, prompts):
		_tokenized = [self.prepare_input(v, self.prompt_text_max_len)  for v in prompts]
		_max_len = max( [ sum(inputs['attention_mask']) for inputs in _tokenized])
		_max_len_toks = [ ]
		for inputs in _tokenized:
			for k, v in inputs.items():
				inputs[k] = inputs[k][:_max_len]
			_max_len_toks.append(inputs)
		return _max_len_toks

	def prepare_input(self, text, max_len):
		inputs = self.tokenizer.encode_plus(
			text,
			return_tensors=None,
			add_special_tokens=True,
			max_length = max_len,
			pad_to_max_length = True,#IF not sort with len,will be True
			truncation = True
		)
		for k, v in inputs.items():
			inputs[k] = torch.tensor(v, dtype=torch.long)
		return inputs

	def __len__(self):
		return len(self.texts)
	def __getitem__(self, _id ):
		inputs = self.prepare_input(self.texts[_id], self.text_max_len)
		prompt_inputs = self.prompt_dict[self.prompt_id[_id]]
		inputs_ids = inputs['input_ids'].numpy()
		mask = np.isin(inputs_ids, self.sentSEPs)

		if self.add_question:
			quesidx = np.where(inputs_ids == self.quesSEP)[0][0]
			mask[:quesidx] = 0
			mask = np.cumsum(mask)
			mask[quesidx] = 0
		else:
			mask = np.cumsum(mask)
			mask[0] = 0
		mask[mask==mask[-1]]=0
		pool_mask = (mask>0)

		#============================
		# mask : head atteion pooling
		# pool_mask : pooling
		#====================

		inputs['slable'] = torch.LongTensor(mask) 
		if self.pooling_name in ['MeanPooling','MaxPooling','MinPooling']:
			inputs['smask'] = torch.LongTensor(pool_mask) 
		elif self.pooling_name=="GeMText":
			inputs['smask'] = inputs["attention_mask"]
		if self.labels is not None:
			inputs['target'] = torch.tensor(self.labels[_id], dtype=torch.float)
		return inputs, prompt_inputs
		

def collate(inputs, prompt_inputs):#, sentmask):
	mask_len = int(inputs["attention_mask"].sum(axis=1).max())
	for k, v in inputs.items():
		inputs[k] = inputs[k][:, :mask_len]
	_mask_len = int(prompt_inputs["attention_mask"].sum(axis=1).max())
	for k, v in prompt_inputs.items():
		prompt_inputs[k] = prompt_inputs[k][:, :_mask_len]
	return (inputs, prompt_inputs)

def batch_to_device(inputs, prompt_inputs, device):
    inputs = {key: inputs[key].to(device) for key in inputs}
    prompt_inputs = {key: prompt_inputs[key].to(device) for key in prompt_inputs}
    return inputs, prompt_inputs