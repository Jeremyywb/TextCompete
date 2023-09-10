from typing import List, Dict, Any, Callable, Optional, Tuple
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import matplotlib.pyplot as plt
import math
import numpy as np
import sys
import torch
import time
import datetime
import copy

from TextCompete.metrics_loss.utils import (
     IntervalStrategy, AverageMeter,ESStrategy,timeSince,LR_HIST
    )

def get_logger(args,LogFileName):
    filename = f"{args.modelRootPath}/{LogFileName}"
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    # handler1 = StreamHandler()
    # handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    # logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


# ======================================================
# EarlyStopping and History
class EarlyStopping:
    def __init__(self, patience=7,max_minze=True, delta=0):
        self._patience = patience
        self._counter = 0
        self.should_training_stop = False
        self._improved = None
        self._MAXIMIZE = max_minze
        self._mode = 1 if self._MAXIMIZE else -1

        #===================================
        # start from worse! IF MAXIMIZE -inf
        self._best_score = -self._mode*np.inf 
        #===================================

        self._delta = delta
    def reset(self):
        self._best_score
    def __call__(self, score):
        self._improved = False
        if self._best_score*self._mode <  score*self._mode :
            print('\n'+"**********"*5+
                f"\n<BEST SCORED {score:.4f} ! IMPROVED FROM {self._best_score:.4f}>\n"+
                "**********"*5
                )
            self._best_score = score
            self._improved = True
            self._counter = 0
        else :
            self._counter += 1
            if self._counter > self._patience:
                self.should_training_stop = True
                print(f'******EarlyStopping: NOIMPROVMENT for {self._counter} steps out of {self._patience}******')



class History(object):
    """Callback that records events into a `History` object.

    Args:
        verbose(int): Print results every verbose iteration.

    Attributes:
        _verbose(int): Print results every verbose iteration.
        _history(Dict[str, Any]): Record all information of metrics of each epoch.
        _start_time(float): Start time of training.
        _epoch_loss(float): Average loss per epoch.
        _epoch_metrics(Dict[str, Any]): Record all information of metrics of each epoch.
        _samples_seen(int): Traversed samples.
    """
    def __init__(
        self,
        args,
        LOGGER,
        verbose: int = 1
    ):
        super(History, self).__init__()
        self._verbose = verbose
        self.args = args
        self._epochs = args.trainer['epochs']
        self.accum_LR = []
        self.logger = LOGGER
        self.foldloss = []
        self.foldeval = []



    def _prepare_args(self, trainloader ):
        
        self.epoch = 0
        self.num_train_steps = len(trainloader)
        self.completed_steps = 0
        if self.args.start_eval_step <1:
            self.start_eval_step = math.ceil( self.args.start_eval_step*self.num_train_steps)
        else:
            self.start_eval_step = self.args.start_eval_step


        #=================================
        # eval_steps reset

        if self.args.eval_steps<1:
            self.eval_steps = math.ceil(self.num_train_steps * self.args.eval_steps)
        else:
            self.eval_steps = self.args.eval_steps
        
        if self.eval_steps>self.args.trainer['gradient_accumulation_steps']:
            coef_dividable_2acc = self.eval_steps//self.args.trainer['gradient_accumulation_steps']
            self.eval_steps = coef_dividable_2acc*self.args.trainer['gradient_accumulation_steps']
        else:
            self.eval_steps = self.args.trainer['gradient_accumulation_steps']
        #=====================================================================

        if (
            self.args.evaluation_strategy == IntervalStrategy.EPOCH.value
            and self.args.es_strategy != ESStrategy.EPOCHS.value
        ):
            print("Strategy conflict!!! evaluation_strategy and es_strategy should meet")
            raise ValueError(
                f'''evaluation_strategy:{self.args.evaluation_strategy}'''
                f'''es_strategy:{self.args.es_strategy}'''
                f'''DOSE NOT MEET!!!'''
            )
        self.num_eval_epoch = math.ceil(self.num_train_steps/self.eval_steps)

        if self.args.es_strategy == ESStrategy.EPOCHS.value:
            patience = 1

        if self.args.es_strategy == ESStrategy.HALF.value:
            patience = math.ceil(self.num_eval_epoch*1/2)

        if self.args.es_strategy == ESStrategy.ONE_THIRD.value:
            patience = math.ceil(self.num_eval_epoch*1/3)

        if self.args.es_strategy == ESStrategy.A_QUARTER.value:
            patience = math.ceil(self.num_eval_epoch*1/4)

        if self.args.es_strategy == ESStrategy.ONE_FIFTH.value:
            patience = math.ceil(self.num_eval_epoch*1/5)
        
        _head = '=========================================='
        _line = '__________________________________________'
        msg = (

             "      TYPE            STEPS\n"
            f'{_head}\n'
            f"     TRAIN             {self.num_train_steps}\n"
            f'{_line}\n'
            f"    NUMEVAL            {self.num_eval_epoch}\n"
            f'{_line}\n'
            f"   STARTEVAL           {self.start_eval_step }\n"
            f'{_line}\n'
            f"EVAL-EVERY-N-STEPS     {self.eval_steps }\n"
            f'{_line}\n'
            f"  PATIENCE-STEPS       {patience} \n"
            f'{_line}\n'
            f"   ESStrategy          {self.args.es_strategy }\n"
            f'{_line}\n'
        )
        if self._verbose==1:
            self.logger.info(msg )
            print(msg)

        return (
                self.num_train_steps,
                self.start_eval_step,
                self.eval_steps,
                patience)

    def _save_checkpoint(
        self, 
        model,
        EARLY_STOPPING
    ):
        if EARLY_STOPPING._improved:
            if self.args.trainer['HEADTrain']:
                torch.save(model.HEAD.state_dict(),self.args.headModel)
                msg = f"***Saving {self.args.headModel.split('/')[-1]}, best score:{EARLY_STOPPING._best_score} ***"
            else:
                torch.save(model.state_dict(),self.args.foldModel)
                msg = f"***Saving {self.args.foldModel.split('/')[-1]}, best score:{EARLY_STOPPING._best_score} ***"
                
            print(msg)
            self.logger.info( msg  )

    def loss_steps_vis(self):
        f, ax = plt.subplots(figsize=(8, 4.5))
        ax.tick_params(color='green', labelcolor='#525B56')
        for spine in ['right', 'top']:
            ax.spines[spine].set_visible(False)
            ax.spines[spine].set_edgecolor('#2A3457')
        ax.grid(which="major", color="lightgrey", linestyle="-",  linewidth=0.5, alpha=0.5)
        plt.title("LOSS History \n")
        plt.plot(range(len(self.foldloss)),self.foldloss,
                 '-o', 
                 color="#4285F4",
                 markeredgecolor='#4285F4',
                 linewidth=2
                 ,markerfacecolor='#4285F4', 
                 label="trainloss")
        plt.plot(range(len(self.foldeval)),self.foldeval,
                 '-o', 
                 color="red",
                 markeredgecolor='red',
                 linewidth=2
                 ,markerfacecolor='red', 
                 label="valloss")
        plt.legend(loc='upper right')
        plt.xlabel("steps")
        plt.ylabel("valloss")
        plt.show()

    def on_train_begin(
        self,
        logs: Optional[Dict[str, Any]] = None
    ):
        """Called at the start of training.

        Args:
            logs(Dict[str, Any]|None): The logs is a dict or None.
        """
        self._history = {"loss": [], "lr": []}
        self._start_time = logs["start_time"]
        self._epoch_loss = 0. # nqa

    def on_epoch_begin(
        self,
    ):
        """Called at the beginning of each epoch.
        """
        #====================================
        # on epoch begin
        self.eval_inner_losses = AverageMeter()
        self.epcoh_start_time = time.time()
        #====================================

        self._epoch_metrics = {"Train Loss": 0.} # nqa
        self.epoch_time_start = time.time()*1000

        self.UNtrained_INEpoch_steps = self.num_train_steps
        self.epoch += 1

        self.eval_interval_grad_norms = {"accum-LR":[], "GradNorm":[], "PreGradNorm":[]}
        self.gradient_accumulation_steps = (
            self.args.trainer['gradient_accumulation_steps']
            if self.args.trainer['use_accumulation']
            else 1
        )

        

        return self.gradient_accumulation_steps


    def _reset_to_next_eval(self):
        if self._verbose==1:
            print(f"\nEpoch {self.epoch}/{self._epochs}")
        self.toNex_start_time = time.time()
        self.eval_inner_losses.reset()
        self.since_last_evaled_step = 0
        self.eval_interval_grad_norms = {"accum-LR":[], "GradNorm":[], "PreGradNorm":[]}

    def on_next_eval(self,step,msg):
        self.foldloss.append(msg['glbloss'])
        self.foldeval.append(msg['valloss'])
        self.accum_LR += self.eval_interval_grad_norms['accum-LR']
        LR = self.accum_LR[-1]
        if self._verbose==0:
            return
        if self.eval_interval_grad_norms['GradNorm']:
            stack_to_np = torch.tensor(self.eval_interval_grad_norms['GradNorm'])
            # print("+++++++++++++++++++stack_to_np+++++++++++++++++++")
            # print(stack_to_np)
            stack_to_np2 = torch.tensor(self.eval_interval_grad_norms['PreGradNorm'])
            
            if torch.logical_or(stack_to_np.isnan().any(),stack_to_np.isinf().any()):
                MaxGrad = "None"
                AvgGrad = "None"
                PreAvgGrad = "None"
            else:
                MaxGrad = int(stack_to_np.cpu().numpy().max())
                AvgGrad = stack_to_np.cpu().numpy().mean()
                if len(str(int(AvgGrad)))<4:
                    AvgGrad = str(AvgGrad)[:4]
                else:
                    AvgGrad = int(AvgGrad)
                
                PreAvgGrad = int(stack_to_np2.cpu().numpy().max())                
        else:
            MaxGrad = 'no'
            AvgGrad = 'no'
            PreAvgGrad = 'no'

        interval_msg = (
                # '{0}/{1} [==============]'
                ' {remain:s} '
                ' - loss: {loss.val:.4f} '
                ' - maxgrad: {MaxGrad}'
                ' - avggrad: {AvgGrad}'
                ' - pregrad: {PreAvgGrad}'
                ' - lasgrad:{lasgrad}'
                ' - lr: {lr:.2f}e-5'
                .format(
                      # step+1, self.num_train_steps,
                      remain=timeSince(self.toNex_start_time, float(step+1)/self.num_train_steps),
                      loss=self.eval_inner_losses,
                      MaxGrad=MaxGrad,
                      AvgGrad=AvgGrad,
                      lasgrad=self.eval_interval_grad_norms['GradNorm'][-1],
                      PreAvgGrad=PreAvgGrad,
                      lr=LR*10000)
            )
        
        self.logger.info(f"\nEpoch {self.epoch}/{self._epochs}")
        self.logger.info(interval_msg)
        print(interval_msg)

        # ====================================================
        # proceding line
        j = (step + 1) / self.num_train_steps
        mid = ">" if step < self.num_train_steps-1 else '='
        proced = int(30*j)
        line = '='*(proced-1)+mid+ '.'*( 30-proced )
        spaces = " " * (len(str(self.num_train_steps)) - len(str(step + 1)))
        # ====================================================

        eval_step_msg = (
                '{0}/{1}{2} [{3}]'
                .format(step+1, self.num_train_steps,spaces,line))
        start_value = None
        for metric_name,metric_value in msg.items():
            if start_value is None:
                start_value = metric_name[0].upper()
                eval_step_msg += f' {metric_name}: {metric_value:.4f}'
            elif start_value != metric_name[0].upper():
                start_value = metric_name[0].upper()
                eval_step_msg += f'\n{step+1}/{self.num_train_steps}{spaces} [{line}] {metric_name}: {metric_value:.4f}'
            else:
                eval_step_msg += f' - {metric_name}: {metric_value:.4f}'
        self.logger.info(eval_step_msg)
        print(eval_step_msg)



    def on_accumulation_end(self,accumulation_step_msg):
        self.eval_interval_grad_norms['accum-LR'].append(accumulation_step_msg['accum-LR'])
        self.eval_interval_grad_norms['PreGradNorm'].append(accumulation_step_msg['PreGradNorm'])
        if 'GradNorm' in accumulation_step_msg:
            self.eval_interval_grad_norms['GradNorm'].append(accumulation_step_msg['GradNorm'])
                
        
        
    def on_epoch_end(
        self,
    ):
        if self._verbose==1:
            # LR_HIST(self.accum_LR)
            self.loss_steps_vis()
        


    def on_step_end(
        self,
        loss,
        target
    ):
        """Called at the end of each batch in training.

        Args:
            batch(int): The index of batch.
            logs(Dict[str, Any]|None): The logs is a dict or None.
                contains `loss` and `batch_size`.
        """

        self.UNtrained_INEpoch_steps -= 1
        self.completed_steps +=1
        self.since_last_evaled_step += 1

        #====================================
        #on step end
        batch_size = target.size(0)
        self.eval_inner_losses.update(loss.item(), batch_size)

        #=====================================================
