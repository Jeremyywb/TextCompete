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


def _logger(versionName=versionName):
	 from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
	 logger = getLogger(__name__)
	 logger.setLevel(INFO)
	 handler1 = StreamHandler()
	 handler1.setFormatter(Formatter("%(message)s"))
	 handler2 = FileHandler(versionName=f"{filename}.log")
	 handler2.setFormatter(Formatter("%(message)s"))
	 logger.addHandler(handler1)
	 logger.addHandler(handler2)
	 return logger


def PathSet(args):
	if args.platform['isgoogle']:
		args.checkpoints_path = args.platform['google']['opath']
		args.DPath = args.platform['google']['dpath']
	else:
		args.checkpoints_path = args.platform['featurize']['opath']
		args.DPath = args.platform['featurize']['dpath']
	if not os.path.exists(args.checkpoints_path):
		os.makedirs(args.checkpoints_path)
	return args

def load_files(args):
	summary_df = pd.read_csv(f'{args.DPath}summaries_train.csv')
	prompt_df =  pd.read_csv(f'{args.DPath}prompts_train.csv')
	return summary_df, prompt_df


# cfg_path = '...'
# with open(cfg_path, 'r') as f:
# 	args = yaml.safe_load(f)
# args = SimpleNamespace(**args)
# args = PathSet(args)
# summary_df, prompt_df = load_files(args)
# processor = PreProcess(**args.data['prepare'])
# train_logger = _logger(versionName=args.checkpoints_path + 'train')
# ver_logger = _logger(versionName=args.checkpoints_path + 'modelVersions')
# summary_df, prompt_df = processor.processor(summary_df, prompt_df)
# kfold(args,summary_df, prompt_df, train_logger, ver_logger)