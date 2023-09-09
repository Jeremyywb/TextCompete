from enum import Enum, unique
import torch
from torch import nn, optim
from collections.abc import Iterable
import matplotlib.pyplot as plt
import time
import math
import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional

import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

# ==========================
# model evaluate strategy
@unique
class IntervalStrategy(Enum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
# ==========================




# ==========================
# model evaluate strategy
@unique
class ESStrategy(Enum):
    EPOCHS = 'epochs'
    HALF = "half"
    ONE_THIRD = "one_third"
    A_QUARTER = "a_quarter"
    ONE_FIFTH = "one_fifth"
# ==========================



#===========================================================
# AverageMeter
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
#===========================================================

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (R %s)' % (asMinutes(s), asMinutes(rs))


# ==========================
def LR_HIST(Lrlist):
    plt.figure(figsize=(8,4))
    plt.title("LR SCHEDULE\n")
    plt.plot(range(len(Lrlist)),Lrlist)
    plt.xlabel("steps")
    plt.ylabel("LR")
    plt.show()


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5

# optim.param_groups
class AGC(optim.Optimizer):
    """Generic implementation of the Adaptive Gradient Clipping

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
      model (torch.nn.Module, optional): The original model
      ignore_agc (str, Iterable, optional): Layers for AGC to ignore
    """

    def __init__(self, 
        # clip_params,
        # no_clip_params,
        ignore_head, 
        optim: optim.Optimizer, 
        clipping: float = 1e-2, 
        eps: float = 1e-3, model=None, ignore_agc=["fc"]):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}

        # if not isinstance(ignore_agc, Iterable):
        #     ignore_agc = [ignore_agc]
        # params = []

        # if model is not None:
            # assert ignore_agc not in [
            #     None, []], "You must specify ignore_agc for AGC to ignore fc-like(or other) layers"
            # names = [name for name, module in model.named_modules()]

            # for module_name in ignore_agc:
            #     if module_name not in names:
            #         raise ModuleNotFoundError(
            #             "Module name {} not found in the model".format(module_name))
        #     params = [{"params": list(module.parameters())} for name,
        #                   module in model.named_modules() if name not in ignore_agc]
        
        # else:
        #     params = [{"params": params}]

        self.ignore_head = ignore_head
        # if ignore_head:
        #     self.clip_params = clip_params
        # else:
        #     self.clip_params = clip_params+no_clip_params
        self.agc_eps = eps
        self.clipping = clipping
        
        self.param_groups = optim.param_groups
        self.state = optim.state

    def unit_norm(self, x):
        """ axis-based Euclidean norm"""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple(
                [x for x in range(1, xlen)]
            )  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    # copy from ranger21
    #https://github.com/lessw2020/Ranger21/blob/main/ranger21/ranger21.py#L415
    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.
        the hardcoded 1e-6 is simple stop from div by zero and no relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        p_norm = self.unit_norm(p).clamp_(self.agc_eps)
        g_norm = self.unit_norm(p.grad)

        max_norm = p_norm * self.clipping

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self.ignore_head:
                if group['ClipGroupName'] == 'head':
                    continue
            for p in group['params']:
                if p.grad is None:
                    continue
                self.agc(p)

                # param_norm = torch.max(unitwise_norm(
                #     p.detach()), torch.tensor(self.agc_eps).to(p.device))
                # grad_norm = unitwise_norm(p.grad.detach())
                # max_norm = param_norm * self.clipping

                # trigger = grad_norm > max_norm

                # clipped_grad = p.grad * \
                #     (max_norm / torch.max(grad_norm,
                #                           torch.tensor(1e-6).to(grad_norm.device)))
                # p.grad.detach().data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """Sets the gradients of all optimized :class:`torch.Tensor` s to zero.

        Arguments:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This is will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                # if 'params' in p:
                #     p = p['params']
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()

def normalize_gradient(x, use_channels=False, epsilon=1e-8):
    """  use stdev to normalize gradients """
    size = x.dim()
    # print(f"size = {size}")

    if (size > 1) and use_channels:
        s = x.std(dim=tuple(range(1, size)), keepdim=True) + epsilon
        # print(f"s = {s}")
        x.div_(s)  # , keepdim=True)

    elif torch.numel(x) > 2:
        s = x.std() + epsilon
        x.div_(s)  # , keepdim=True)
    return x



class CustomAGC:
    def __init__(self,eps=1e-3,clipping=1e-2, grad_norm=False):
        self.paramter_eps = eps
        self.clipping = clipping

    def unit_norm(self, x):
        """ axis-based Euclidean norm"""
        # verify shape
        keepdim = True
        dim = None

        xlen = len(x.shape)
        # print(f"xlen = {xlen}")

        if xlen <= 1:
            keepdim = False
        elif xlen in (2, 3):  # linear layers
            dim = 1
        elif xlen == 4:  # conv kernels
            dim = (1, 2, 3)
        else:
            dim = tuple(
                [x for x in range(1, xlen)]
            )  # create 1,..., xlen-1 tuple, while avoiding last dim ...

        return x.norm(dim=dim, keepdim=keepdim, p=2.0)

    def agc(self, p):
        """clip gradient values in excess of the unitwise norm.
        the hardcoded 1e-6 is simple stop from div by zero and no relation to standard optimizer eps
        """

        # params = [p for p in parameters if p.grad is not None]
        # if not params:
        #    return

        # for p in params:
        p_norm = self.unit_norm(p).clamp_(self.paramter_eps)
        g_norm = self.unit_norm(p.grad)

        max_norm = p_norm * self.clipping

        clipped_grad = p.grad * (max_norm / g_norm.clamp(min=1e-6))

        new_grads = torch.where(g_norm > max_norm, clipped_grad, p.grad)
        p.grad.detach().copy_(new_grads)

    def clip_grad_adptive_(self,
            parameters: _tensor_or_tensors):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            if p.grad is None:
                continue
            self.agc(p)
        # grads = [p.grad for p in parameters if p.grad is not None]
        # if len(grads) == 0:
        #     return torch.tensor(0.)
        # first_device = grads[0].device
        # grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        #     = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]
        # for ((device, _), [grads]) in grouped_grads.items():
        #     for g in grads:
        #         self.agc(g)




def calcu_grad_norm(
        parameters: _tensor_or_tensors, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor:
    total_norm = 0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


class ModelSummary:
    def __init__(self, model, input_size,device):
        self.device = device
        self.model = model
        self.input_size = input_size
        self.hooks = []

    def forward_hook(self, module, input, output, name):
        if 'backbone' in name:
          return
        input_shapes = [f"{list(i.shape)}" for i  in input if i is not None]
        input_shapes = ', '.join(input_shapes)
        num_params = sum(p.numel() for p in module.parameters())
        output_shape = list(output[0].shape)
        self.print_row(name, module.__class__.__name__, input_shapes, output_shape, num_params)

    def print_row(self, name, layer, input_shapes, output_shape, num_params):
        print(f"{name} ({layer})".ljust(25),
              "|", input_shapes.ljust(15).center(30), "|",
              str(output_shape).ljust(15).center(30), "|", num_params)

    def backward_hook(self, module, grad_input, grad_output):
        pass

    def summary(self):
        summlen = 110
        print("=" * summlen)
        print("     Model Summary:")
        print("=" * summlen)
        self.print_row('Layer', 'Type', 'Input Shapes', 'Output Shape', 'Param #')
        print("=" * summlen)

        for layer_name, layer in self.model.named_children():
            hook = layer.register_forward_hook(lambda module, input, output, name=layer_name: self.forward_hook(module, input, output, name))
            self.hooks.append(hook)

        with torch.no_grad():
            input_tensor = [torch.randint(1000,size).to(self.device) if size is not None else None for size in self.input_size]
            self.model(*input_tensor)
        total_params = 0  # 用于统计总参数数量
        trainable_params = 0  # 用于统计可训练参数数量
        non_trainable_params = 0  # 用于统计不可训练参数数量
        for hook in self.hooks:
            hook.remove()
        # 计算总参数数量、可训练参数数量和不可训练参数数量
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                non_trainable_params += param.numel()

        # 打印 "Total params"、"Trainable params" 和 "Non-trainable params" 信息
        print("=" * summlen)
        print(f"Total params: {total_params}({round(total_params /(1024*1024) ,2)}M)".ljust(30))
        print(f"Trainable params: {trainable_params}({round(trainable_params /(1024*1024) ,2)}M)".ljust(30))
        print(f"Non-trainable params: {non_trainable_params}({round(non_trainable_params /(1024*1024) ,2)}M)".ljust(30))
        print("=" * summlen)

        # print("=" * summlen)
