# ====================================================
# Directory settings
# ====================================================


import sys

# . /accelerate/start
# kaggle datasets version -p /content/gdrive/MyDrive/output --dir-mode zip
# ln -s /root/miniconda3/lib/python3.10/site-packages/torch/lib/libnvrtc-672ee683.so.11.2 /root/miniconda3/lib/python3.10/site-packages/torch/lib/libnvrtc.so

# sys.path.append('home/ModelRoot')
# sys.path.append('home/ModelRoot/W_Informer')
# export TOKENIZERS_PARALLELISM=true
# export PYTHONPATH="${PYTHONPATH}:/home/ModelRoot"
# export PYTHONPATH="${PYTHONPATH}:/home/ModelRoot/W_Informer"
# source ~/.bashrc
# sys.path.append('/home/featurize/work/Libs')
# os.system('pip uninstall -y transformers')


# ====================================================
# Library
# ====================================================

import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")

import scipy as sp
import numpy as np
import pandas as pd
from math import sqrt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold


from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
from torch.nn import Parameter
from dataclasses import dataclass
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast


# os.system('python -m pip install --no-index --find-links=../input/fb3-pip-wheels tokenizers')
import tokenizers
import transformers
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# %env TOKENIZERS_PARALLELISM=true

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy
# assert tokenizers.__version__=='0.13.3'
# assert transformers.__version__=='4.30.2'
# assert scipy.__version__ == '1.7.3'
import os

os.environ['CURL_CA_BUNDLE'] = ''
import copy
import optuna

# same with kaggle version, if train outside

# tokenizers.__version__: 0.13.3
# transformers.__version__: 4.30.2
# env: TOKENIZERS_PARALLELISM=false



import yaml 
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
from TextCompete.data_utils.preprocess import PreProcess
from TextCompete.train_utils import kfold

# ======================
#  NOTEBOOK
# ======================


#=======================
# model version logger
#=======================


def _logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_all(seed: int) -> None:
    """Seed current experiment to guarantee reproducibility.

    Parameters:
        seed: manually specified seed number

    Return:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running with cudnn backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


# pip install optuna
# pip install spacy
# pip install matplotlib
# pip install sentencepiece
# pip install protobuf==3.20.3
# python -m spacy download en_core_web_sm
# pip uninstall -y tokenizers
# pip install transformers==4.30.2
# pip install pandas==1.5.3
# pip install tokenizers==0.13.3
# pip uninstall -y scipy
# pip install scipy==1.7.3
# pip install iterative-stratification==0.1.7
# pip install seaborn

def PathSet(args):
    if args.platform['isgoogle']:
        args.checkpoints_path = args.platform['google']['opath']
        args.DPath = args.platform['google']['dpath']
    else:
        args.checkpoints_path = args.platform['featurize']['opath']
        args.DPath = args.platform['featurize']['dpath']
    if not os.path.exists(args.checkpoints_path):
        os.makedirs(args.checkpoints_path)
    args.version_log = args.checkpoints_path + 'VERSIONS/'
    if not os.path.exists(args.version_log):
        os.makedirs(args.version_log)
    return args

def load_files(args):
    summary_df = pd.read_csv(f'{args.DPath}summaries_train.csv')
    prompt_df =  pd.read_csv(f'{args.DPath}prompts_train.csv')
    return summary_df, prompt_df
def text_len(text):
    texts = text.split("\n")
    return sum([ len(te.split()) for te in  texts])

def objective(trial):
    # Define the hyperparameters to tune
    # 'weight_decay': 0.005, 'LR': 3e-05, 'HeadLR': 0.0001, 'LLDR': 0.9, 'freezing': 4,
    newargs = copy.deepcopy(args)
    weight_decay = trial.suggest_float('weight_decay', 0.01, 0.1)
    LR       = trial.suggest_int('LR', 2, 6)
    HeadLR   = trial.suggest_int('HeadLR', 2, 6)
    LLDR     = trial.suggest_categorical('LLDR', [0.7,0.75,0.8,0.85,0.9])
    freezing = trial.suggest_int('freezing', 6, 22)#large
    newargs.optimizer[OptimName]['weight_decay'] = weight_decay
    newargs.optimizer[OptimName]['lr'] = LR/1e5
    newargs.optimizer['HeadLR'] = HeadLR/1e5
    newargs.optimizer['LLDR'] = LLDR
    newargs.model['params']['freezing'] = freezing

    
    # Create an XGBoost classifier
    version_msg,ver_log_met = kfold(newargs,summary_df, prompt_df,verbose=0)
    score = ver_log_met['oofloss']
    
    return score

import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description='示例脚本，接受带参数名称的参数')

# 添加参数
parser.add_argument('--folds', type=str, help='select train folds:0,1,2,3')
# parser.add_argument('--arg2', type=float, help='第二个参数')
# parser.add_argument('--arg3', type=str, help='第三个参数')

# 解析命令行参数



if __name__ == '__main__':
    # main()
    inputargs = parser.parse_args()
    cfg_path = '/home/ModelRoot/TextCompete/CFG/deberta-v3-large_addPMT.yaml'
    with open(cfg_path, 'r') as f:
        args = yaml.safe_load(f)

    args = SimpleNamespace(**args)
    # seed_everything(args.seed)
    seed_all(args.seed)


    # # =====================================
    # #                  1️⃣
    # # step eval start with half epoch
    # # eval steps: 0.1*epcoh_train_steps
    # args.evaluation_strategy = 'steps'
    # args.es_strategy = 'half'

    # # =====================================
    # #                  2️⃣
    # # es stop at each epoch, iter all epochs

    # args.es_strategy = 'one_third'
    # args.FullEpochStepEval = True

    # # =====================================
    # #                  3️⃣
    # # accumulation n step -->lower train batch_size
    # # gradient_accumulation_steps*batch_size=128
    # # batch_size=16,gradient_accumulation_steps=8

    # # args.trainer['gradient_accumulation_steps'] = 8
    # # args.train_loader['batch_size'] = 16

    # ====================================================================
    # output
    args.save_name_prefix = 'AddPRT'
    args.platform['featurize']['opath'] = '/home/output/'
    args.platform['featurize']['dpath'] = '/home/data/'
    args.modelRootPath = '/home/output/'


    # ====================================================================
    args = PathSet(args)
    folds = inputargs.folds.split(',')
    args.selected_folds = [int(i) for i in folds]



    # #============================================
    # # PROMPT QUESTION
    # args.data['dataset']['pool_question'] = False##
    # args.data['dataset']['pool_middle_sep'] = False##
    # args.data['prepare']['add_question'] = False
    # #============================================

    # #============================================
    # # CLIPPING
    # args.clipgrad['clipname'] = 'clipnorm'
    # args.clipgrad['clipnorm']['max_norm'] = 10

    # #============================================
    # args.model['pooling_params']['pooling_name'] = 'MeanPoolingA'##
    # # args.model['params']['REINIT_LAYERS'] = 1


    # OptimName = args.optimizer['name']
    # schedName = args.scheduler['name']
    # #============================================================================================================================================
    # # hyperparameters
    # hyperparameters = {'weight_decay': 0.005, 'LR': 3e-05, 'HeadLR': 0.0001, 'LLDR': 0.9, 'freezing': 4, 'warmup': 0.0, 'headname': 'dense_01'}
    # args.optimizer[OptimName]['weight_decay'] = hyperparameters['weight_decay']
    # args.optimizer[OptimName]['lr'] = hyperparameters['LR']
    # args.optimizer['HeadLR'] = hyperparameters['HeadLR']
    # args.optimizer['LLDR'] = hyperparameters['LLDR']
    # args.model['params']['freezing'] = hyperparameters['freezing']
    # args.scheduler['warmup'] = hyperparameters['warmup']
    # args.model['params']['headname'] = hyperparameters['headname']
    # #============================================================================================================================================
    # # CV# oofloss: 0.5670710206031799# Ocontent: 0.497537761926651# Owording: 0.6366042494773865

    # #==================================================
    # # args changing
    # # args.clipgrad['CustomAGC']['clipping'] = 0.08
    # args.optimizer[OptimName]['lr'] = hyperparameters['LR']*(24/32)**0.5
    # args.optimizer['HeadLR'] = hyperparameters['HeadLR']*(24/32)**0.5
    # #==================================================

    # #==================================================
    # # mainly of trial
    # args.model['params']['add_prompt'] = True
    # args.train_loader['batch_size'] = 16
    # args.val_loader['batch_size'] = 16
    # # args.selected_folds = [0,1,2,3]
    # #==================================================
    # args.verbose = 1
    # args.data['prepare']['experiment'] = False
    # args.data['prepare']['experiment_rate'] = 0.3


    # =====================================
    #                  1️⃣
    # step eval start with half epoch
    # eval steps: 0.1*epcoh_train_steps
    args.evaluation_strategy = 'steps'
    args.es_strategy = 'half'

    # =====================================
    #                  2️⃣
    # es stop at each epoch, iter all epochs

    # args.es_strategy = 'one_third'
    # args.FullEpochStepEval = True

    # =====================================
    #                  3️⃣
    # accumulation n step -->lower train batch_size
    # gradient_accumulation_steps*batch_size=128
    # batch_size=16,gradient_accumulation_steps=8

    # args.trainer['gradient_accumulation_steps'] = 8
    # args.train_loader['batch_size'] = 16

    # ====================================================================
    # output
    # args.save_name_prefix = 'AddPRT'
    # args.platform['featurize']['opath'] = '/home/featurize/work/output/'
    # args.modelRootPath = '/home/featurize/work/output/'
    # # ====================================================================


    #============================================
    # PROMPT QUESTION
    args.data['dataset']['pool_question'] = True##
    args.data['dataset']['pool_middle_sep'] = True##
    args.data['prepare']['add_question'] = True
    #============================================

    #============================================
    # CLIPPING
    args.clipgrad['clipname'] = 'CustomAGC'
    args.clipgrad['CustomAGC']['clipping'] = 0.01
    #============================================

    # args.selected_folds = [0,1,2,3]
    args.model['pooling_params']['pooling_name'] = 'MeanPoolingA'##
    args.model['params']['REINIT_LAYERS'] = 1


    OptimName = args.optimizer['name']
    schedName = args.scheduler['name']
    #============================================================================================================================================
    # hyperparameters
    hyperparameters = {'weight_decay': 0.005, 'LR': 3e-05, 'HeadLR': 0.0001, 'LLDR': 0.9, 'freezing': 4, 'warmup': 0.0, 'headname': 'dense_01'}
    args.optimizer[OptimName]['weight_decay'] = hyperparameters['weight_decay']
    args.optimizer[OptimName]['lr'] = hyperparameters['LR']
    args.optimizer['HeadLR'] = hyperparameters['HeadLR']
    args.optimizer['LLDR'] = hyperparameters['LLDR']
    args.model['params']['freezing'] = hyperparameters['freezing']
    args.scheduler['warmup'] = hyperparameters['warmup']
    args.model['params']['headname'] = hyperparameters['headname']
    #============================================================================================================================================
    # CV# oofloss: 0.5670710206031799# Ocontent: 0.497537761926651# Owording: 0.6366042494773865

    #==================================================
    # args changing
    args.clipgrad['CustomAGC']['clipping'] = 0.008
    args.optimizer[OptimName]['lr'] = hyperparameters['LR']*(16/32)**0.5
    args.optimizer['HeadLR'] = hyperparameters['HeadLR']*(16/32)**0.5
    #==================================================

    #==================================================
    # mainly of trial
    args.model['params']['add_prompt'] = True
    args.train_loader['batch_size'] = 16
    args.val_loader['batch_size'] = 16
    # args.selected_folds = [0,1,2,3]
    #==================================================
    args.verbose = 0
    args.data['prepare']['experiment'] = False
    args.data['prepare']['experiment_rate'] = 0.3
    args.Sampler ='RandomSampler'

    # ===================
    # processor



    summary_df, prompt_df = load_files(args)
    processor = PreProcess(**args.data['prepare'])
    summary_df, prompt_df = processor.processor(summary_df, prompt_df)
    id2fold = {
        "814d6b": 0,
        "39c16e": 1,
        "3b9047": 2,
        "ebad26": 3,
    }
    summary_df['fold'] = summary_df['prompt_id'].map(id2fold)
    summary_df['textlen'] = summary_df['text'].map(text_len)
    summary_df = summary_df.sort_values(['textlen'],ascending=False).reset_index(drop=True)
    del summary_df['textlen']
    # version_msg,ver_log_met = kfold(args,summary_df, prompt_df,verbose=1)

    # Use Optuna to tune the hyperparameters
    study = optuna.create_study()
    study.optimize(objective, n_trials=10,n_jobs=5)
    
    # Print the best hyperparameters and the best score
    print("Best hyperparameters: ", study.best_params)
    print("Best score: ", study.best_value)

