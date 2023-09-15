import torch
import torch.nn as nn
import numpy as np
from TextCompete.basemodel.heads import *
from torch.cuda.amp import autocast

class AUTOCONVHEAD1(CONVHEAD1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCONVHEAD1, self).__init__(*args, **kwargs)
    
    @autocast()
    def forward(self, x):
        output = self.fc(x.unsqueeze(-1)).squeeze(-1)
        var = torch.exp(self.varlayer(output))
        return (output,var)

class AUTODENSEHEAD1(DENSEHEAD1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTODENSEHEAD1, self).__init__(*args, **kwargs)

    @autocast()
    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)


class AUTOUniFormCONVHEAD1(UniFormCONVHEAD1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOUniFormCONVHEAD1, self).__init__(*args, **kwargs)

    @autocast()
    def forward(self, x):
        output = self.fc(x.unsqueeze(-1)).squeeze(-1)
        var = torch.exp(self.varlayer(output))
        return (output,var)

class AUTOUniFormDENSEHEAD1(UniFormDENSEHEAD1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOUniFormDENSEHEAD1, self).__init__(*args, **kwargs)

    @autocast()
    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)


class AUTOBINSSOFTMAX(BINSSOFTMAX):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOBINSSOFTMAX, self).__init__(*args, **kwargs)

    @autocast()
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



class AUTOBNDENSEHEAD1(BNDENSEHEAD1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOBNDENSEHEAD1, self).__init__(*args, **kwargs)

    @autocast()
    def forward(self, x):
        output = self.fc(x)
        var = torch.exp(self.varlayer(output))
        return (output,var)


class AUTOBINSOFTMAX2CONV1D(BINSOFTMAX2CONV1D):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOBINSOFTMAX2CONV1D, self).__init__(*args, **kwargs)

    @autocast()
    def forward(self, x):
        output = self.fc(x)
        content =  torch.softmax(output[:,:self.numbins],dim=-1) *self.values_bins.unsqueeze(0)
        wroding = torch.softmax(output[:,self.numbins:],dim=-1) *self.values_bins.unsqueeze(0)
        output = torch.cat([
                content.sum(-1).unsqueeze(-1),
                wroding.sum(-1).unsqueeze(-1)
            ],dim=-1)
        output = output/2 + self.conv(x.unsqueeze(-1)).squeeze(-1)/2
        var = torch.exp(self.varlayer(output))
        return (output,var)