import torch
import torch.nn as nn
import numpy as np


class CONVHEAD1(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(CONVHEAD1, self).__init__()
        self.config = config
        self.fc = nn.Sequential(
                nn.Conv1d(finaldim, finaldim//2, 1),
                nn.ELU(),
                nn.Conv1d(finaldim//2, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, output_dim, 1)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)
        if init_head:
            self.fc.apply(self._kaimin)
            self.varlayer.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _kaimin(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        output = self.fc(x.unsqueeze(-1)).squeeze(-1)
        var = torch.exp(self.varlayer(output))
        return (output,var)

class DENSEHEAD1(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(DENSEHEAD1, self).__init__()
        self.config = config
        self.fc = nn.Sequential(
                nn.Linear(finaldim, finaldim//2),
                nn.ELU(),
                nn.Linear(finaldim//2, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)
        if init_head:
            self.fc.apply(self._xavier_init)
            self.varlayer.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)






class UniFormCONVHEAD1(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(UniFormCONVHEAD1, self).__init__()
        self.config = config
        self.fc = nn.Sequential(
                nn.Conv1d(finaldim, finaldim//2, 1),
                nn.ELU(),
                nn.Conv1d(finaldim//2, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, output_dim, 1)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)
        if init_head:
            self.fc.apply(self._kaimin)
            self.varlayer.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def _kaimin(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        output = self.fc(x.unsqueeze(-1)).squeeze(-1)
        var = torch.exp(self.varlayer(output))
        return (output,var)

class UniFormDENSEHEAD1(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(UniFormDENSEHEAD1, self).__init__()
        self.config = config
        self.fc = nn.Sequential(
                nn.Linear(finaldim, finaldim//2),
                nn.ELU(),
                nn.Linear(finaldim//2, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)
        if init_head:
            self.fc.apply(self._xavier_init)
            self.varlayer.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)




class BINSSOFTMAX(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(BINSSOFTMAX, self).__init__()
        self.config = config
        self.numbins = 16
        self.values_bins = nn.Parameter(
            torch.tensor(np.arange(self.numbins)/self.numbins*7.5-2,dtype=torch.float32),
            requires_grad=False
        )
        
        self.fc = nn.Sequential(
                nn.Linear(finaldim, finaldim//2),
                nn.ELU(),
                nn.Linear(finaldim//2, 64),
                nn.ReLU(),
                nn.Linear(64, self.numbins*output_dim)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)

        if init_head:
            self.fc.apply(self._xavier_init)
            self.varlayer.apply(self._xavier_init)

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        output = self.fc(x)
        content =  torch.softmax(output[:,:self.numbins],dim=-1) *self.values_bins.unsqueeze(0)
        wroding = torch.softmax(output[:,self.numbins:],dim=-1) *self.values_bins.unsqueeze(0)
        output = torch.cat([
                content.sum(-1).unsqueeze(-1),
                wroding.sum(-1).unsqueeze(-1)
            ],dim=-1)
        var = torch.exp(self.varlayer(output))
        return (output,var)



class BNDENSEHEAD1(nn.Module):
    def __init__(self, finaldim,output_dim,init_head, config):
        super(BNDENSEHEAD1, self).__init__()
        self.config = config
        self.fc = nn.Sequential(
                nn.Linear(finaldim, finaldim//2),
                nn.BatchNorm1d(finaldim//2),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Linear(finaldim//2, 64),
                nn.ELU(),
                nn.Dropout(0.1),
                nn.Linear(64, output_dim)
            )
        self.varlayer = nn.Linear(output_dim,output_dim)
        if init_head:
            self.fc.apply(self._init_weights)
            self.varlayer.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)
