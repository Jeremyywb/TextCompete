import torch
from torch import nn
from torch.distributions import Categorical
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import math
from sklearn.metrics import mean_squared_error



class RMSELoss(nn.Module):
    def __init__(self, beta,reduction='none', eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class SmoothL1HuberLoss(nn.Module):
    '''
    beta: 1.As beta -> 0, Smooth L1 loss converges to L1Loss, 
        while HuberLoss converges to a constant 0 loss. 
        When beta is 0, Smooth L1 loss is equivalent to L1 loss.
        2.As beta -> +∞, Smooth L1 loss converges to a constant 0 loss, while HuberLoss converges to MSELoss
        3.For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant slope of 1. For HuberLoss, the slope of the L1 segment is beta.
    '''
    def __init__(self, beta=1.0, reduction='mean',eps=None):
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        num_columns = input.size(1)  # 获取列数
        loss_list = []

        for i in range(num_columns):
            input_column = input[:, i]
            target_column = target[:, i]

            n = torch.abs(input_column - target_column)
            cond = n < self.beta

            loss_column = torch.where(cond, 0.5 * n**2 / self.beta, n - 0.5 * self.beta)

            if self.reduction == 'mean':
                loss_list.append(loss_column.mean() if loss_column.numel() > 0 else 0.0 * loss_column.sum())
            elif self.reduction == 'sum':
                loss_list.append(loss_column.sum())
            else:
                loss_list.append(loss_column)

        return torch.stack(loss_list)



class CRMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, y_true, y_pred):
        scores = []
        num_columns = y_true.size(1)  # 获取列数
        for i in range(num_columns):
            y_true_column = y_true[:, i]
            y_pred_column = y_pred[:, i]
            mse = torch.mean((y_true_column - y_pred_column) ** 2)  # 计算均方误差
            rmse = torch.sqrt(mse+self.eps)  # 计算均方根误差
            scores.append(rmse)
        return torch.stack(scores) # 计算均值


class CommonLitLoss(nn.Module):
    def __init__(
        self, 
        loss_name='RMSELoss',
        loss_param = {},
        reduction="mean",weights=None
        ):
        super().__init__()
        self.loss_func = eval(loss_name)(**loss_param)
        self.eps = 1e-6
        self.reduction = reduction
        self.weights = torch.tensor(weights).to(device) if weights else None

    def forward(self, y_pred, y_true):
        loss = self.loss_func(y_pred, y_true)
        if self.weights is not None:
            loss = loss * self.weights
            if self.reduction == 'sum':
                loss = loss.sum()
            else:
                loss = loss.sum() / self.weights.sum()
        else:
            if self.reduction == 'sum':
                loss = loss.sum()
            else:
                loss = loss.mean()
        return loss


class CommonLitCRMSELoss(CommonLitLoss):
    def __init__(
        self,
        loss_name='CRMSELoss',
        loss_param = {},
        reduction="mean",weights=None):
        super(CommonLitCRMSELoss,self).__init__(
            loss_name=loss_name,
            loss_param =loss_param,
            reduction=reduction ,
            weights=weights
        )

class CommonLitHuber(CommonLitLoss):
    def __init__(
        self,
        loss_name='SmoothL1HuberLoss',
        loss_param = {},
        reduction="mean",weights=None):
        super(CommonLitHuber,self).__init__(
            loss_name=loss_name,
            loss_param =loss_param,
            reduction=reduction ,
            weights=weights
        )

# ==========================================
# numpy calculate loss

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(args, _name,y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    met = {_name:mcrmse_score}
    for i,col in enumerate(args.model['target_cols']):
        met[f'{_name[0].upper()}{col}'] = scores[i]
    return met



ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu




def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


class MDNLoss(nn.Module):
    def __init__(self):
        super(MDNLoss, self).__init__()

    def gaussian_probability(self,sigma, mu, target):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.

        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.unsqueeze(1).expand_as(sigma)
        ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
        return torch.prod(ret, 2)

    def forward(self, pi, sigma, mu, target):
        # 在这里实现损失计算逻辑
        prob = pi * self.gaussian_probability(sigma, mu, target)
        nll = -torch.log(torch.sum(prob, dim=1))
        return torch.mean(nll)



def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1)
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn(
        (sigma.size(2), sigma.size(0)), requires_grad=False).to(sigma.device)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return (gaussian_noise * variance_samples + mean_samples).transpose(0, 1)



# from nltk.sentiment.vader import SentimentIntensityAnalyzer

# sentiment_analyzer = SentimentIntensityAnalyzer()

# # 分析总结文本
# sentiments = sentiment_analyzer.polarity_scores(summary)

# # 获取正面和负面词汇
# positive_words = set()
# negative_words = set()

# # 根据情感分数判断词语的情感，并将唯一的正面和负面词汇添加到集合中
# for word in summary.split():
#     word_sentiment = sentiment_analyzer.polarity_scores(word)
#     if word_sentiment['pos'] > 0:
#         positive_words.add(word)
#     elif word_sentiment['neg'] > 0:
#         negative_words.add(word)

# # 输出唯一的正面和负面词汇数量
# print(f"唯一的正面词汇数量: {len(positive_words)}")
# print(f"唯一的负面词汇数量: {len(negative_words)}")


对于服从正态分布的连续变量，这不是问题

Kolmogorov-Smirnov检验评估连续变量的正态性

最优尺度回归


目标分析
特征工程
单变量分析
