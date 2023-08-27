from enum import Enum, unique
import matplotlib.pyplot as plt
import time
import math

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
    HALF = "half"
    ONE_THIRD = "one_third"
    A_QUARTER = "a_quarter"
    ONE_FIFTH = "one_fifth"
    EPOCHS = 'epochs'
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
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# ==========================
def LR_HIST(Lrlist):
    plt.title("LR SCHEDULE\n")
    plt.plot(range(len(Lrlist)),Lrlist)
    plt.xlabel("steps")
    plt.ylabel("LR")
    plt.show()

