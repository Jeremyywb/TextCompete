# =============================
# dateset
# =============================

import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader,RandomSampler


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
        pool_middle_sep,
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
        self.pool_middle_sep = pool_middle_sep
        self.add_question = add_question
        self.pooling_name = pooling_name
        self.multilpool = multilpool
        self.span_pool = span_pool
        self.sentSEPs = ['!','?','.','\n','[CLS]','[SEP]']#CLS:1,SEP:2
        self.sentSEPs = tokenizer.convert_tokens_to_ids(self.sentSEPs)#list
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

            list_prompt_inputs = []
            for prompt_text in self.prompt_df['prompt_text'].values:
                list_prompt_inputs.append(
                    self.prepare_input(prompt_text, self.prompt_text_max_len)
                    )
                
            max_prompt_len = max( [ sum(inp['attention_mask']) for inp in list_prompt_inputs])

            for prompt_id, prompt_input in zip(self.prompt_df['prompt_id'].values, list_prompt_inputs):
                for k, v in prompt_input.items():
                    prompt_input[k] = prompt_input[k][:max_prompt_len]
                self.prompt_dict[prompt_id] = prompt_input

    def prepare_input(self,text, max_len):
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
        return self.samples[_id]

    def _create_samples(self):
        self.samples = []
        for _id in range(len(self.texts)):
            features = []
            inputs = {}
            summary_inputs = self.prepare_input(self.texts[_id], self.text_max_len)
            for k, v in summary_inputs.items():
                inputs["summary_"+k] = summary_inputs[k]
            features.extend(['summary_input_ids','summary_attention_mask','summary_token_type_ids'])
            if self.prompt_dict is not None:
                prompt_inputs = self.prompt_dict[self.prompt_id[_id]]
            else:
                prompt_inputs = None
            inputs_ids = inputs['summary_input_ids'].numpy()
            if self.pool_question and self.pool_middle_sep:
                sepmask = np.isin(inputs_ids,[1,2])
                mask = sepmask.cumsum()
                mask[mask==mask[-1]]=0
                mask[0]=0
                mask= (mask>0)*1
            elif self.pool_question:
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
            elif self.pooling_name in ["GeMText",'MeanPoolingA']:
                inputs['summary_smask'] = inputs["summary_attention_mask"]
                features.append('summary_smask')
            if self.labels is not None:
                inputs['target'] = torch.tensor(self.labels[_id], dtype=torch.float)
                features.append('target')

            if prompt_inputs is not None:
                inputs.update(
                    {'prompt_input_ids':prompt_inputs['input_ids'],
                    'prompt_attention_mask':prompt_inputs['attention_mask']}
                    )
                features.extend(['prompt_input_ids','prompt_attention_mask'])
            self.samples.append(inputs)


def collate(inputs):#, sentmask):
    for intype in ['summary','prompt']:
        DEBUGMSG = f"[collate fuc]: intype: {intype}"
        print(DEBUGMSG)
        attention_mask = f"{intype}_attention_mask"
        DEBUGMSG = f"[collate fuc]: attention_mask: {attention_mask}"
        print(DEBUGMSG)

        if attention_mask in inputs:
            _mask_len = int(inputs[attention_mask].sum(axis=1).max())
            DEBUGMSG = f"[collate fuc]: _mask_len:{_mask_len}"
            print(DEBUGMSG)
            for k in list(inputs):
                if intype in k:
                    DEBUGMSG = f"[collate fuc]: k:{k}"
                    print(DEBUGMSG)
                    inputs[k] = inputs[k][:, :_mask_len]
    return inputs

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict

def get_loader( args, tokenizer,summary_df,prompt_df, fold=None ):
    DEBUGMSG = "============================================="
    DEBUGLINE= "_____________________________________________"
    DEBUGMSG += '\n<get_loader>'
    dset_parameters = args.data['dataset']
    dset_parameters.update(
        {"pooling_name":args.model['pooling_params']['pooling_name'],
         'multilpool' : args.model['params']['multilpool'],
         'span_pool' : args.model['params']['span_pool'],
         'add_question' : args.data['prepare']['add_question'],
        }
    )
    if fold is not None:
        DEBUGMSG += f"\n{str(dset_parameters)}\n{DEBUGLINE}"
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
        # =============================================
        # sampler
        if args.Sampler == 'noo':
            train_loader = DataLoader(train_dataset,**args.train_loader)
            DEBUGMSG += "\ntrain_loader: No sampler\n"
        elif args.Sampler == 'StratifiedSampler':
            args.train_loader['num_workers'] = 0
            args.train_loader['shuffle'] = False
            args.train_loader['drop_last'] = False
            class_V = torch.from_numpy(train_df['fold'].values)
            sampler = StratifiedSampler(class_vector=class_V, batch_size=args.train_loader['batch_size'])
            train_loader = DataLoader(train_dataset,sampler=sampler,**args.train_loader)
            DEBUGMSG += "\ntrain_loader: StratifiedSampler\n"
        elif args.Sampler == 'RandomSampler':
            args.train_loader['shuffle'] = False
            train_loader = DataLoader(train_dataset,sampler=RandomSampler(train_dataset) ,**args.train_loader)
            DEBUGMSG += "\ntrain_loader: RandomSampler\n"
        DEBUGMSG = f"\n{str(args.train_loader)}\n{DEBUGLINE}"
        # =====================================================================

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



class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class StratifiedSampler(Sampler):
    """Stratified Sampling

    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)