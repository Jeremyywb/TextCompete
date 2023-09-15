import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
import torch.utils.checkpoint
from torch.cuda.amp import autocast
import torch.nn.functional as F
from TextCompete.basemodel.pooling import NLPPooling
from W_Informer.models.domian_encoder import PromptAwareEncoder,ConvPoolLayer
from TextCompete.basemodel.autocastheads import *
from TextCompete.basemodel.models import CommonLitModelV1
import TextCompete.metrics_loss.loss_function as mdn
import os


AUTOHEAD_MAPPER = {
    "conv1d_01":AUTOCONVHEAD1,
    'dense_01':AUTODENSEHEAD1,
    'uniform_conv1d_01':AUTOUniFormCONVHEAD1,
    'uniform_dense_01':AUTOUniFormDENSEHEAD1,
    'binssoftmax_01':AUTOBINSSOFTMAX,
    'bndense_01':AUTOBNDENSEHEAD1,
    'binssoftmax2conv1d':AUTOBINSOFTMAX2CONV1D,
}

def top_n_layer_freeze(module,n):
    # for _name,p in module.embeddings.named_parameters():
    #     p.requires_grad = False
    for i in range(0,n,1):
        for _name,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False


class AUTOCommonLitModelV1(CommonLitModelV1):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCommonLitModelV1, self).__init__(*args, **kwargs)
        self.HEAD = AUTOHEAD_MAPPER[self.headname]( 
            finaldim   = self.finaldim,
            output_dim = self.output_dim,
            init_head  = self.init_head,
            config     = self.config  
        )

    @autocast()
    def forward(
        self, 
        summary_input_ids,
        summary_attention_mask,
        summary_token_type_ids,
        summary_slable=None,
        summary_smask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None
                            ):

        hidden_states = self.backbone(
            input_ids=summary_input_ids, 
            attention_mask = summary_attention_mask,
            token_type_ids = summary_token_type_ids 
            )[0]

        del summary_input_ids,summary_attention_mask,summary_token_type_ids
        if self.add_prompt:
            p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]
            hidden_states = self.encoders(
                    hidden_states,
                    p_hidden_states,
                    # prompt_inputs['attention_mask']
                )
            del p_hidden_states,prompt_attention_mask,prompt_input_ids

        if self.multilpool:
            poolout = torch.cat([
                self.pool_ly(hidden_states,summary_smask),
                self.spans_pool(hidden_states,summary_slable),
                 ],dim=-1)
            del summary_slable,summary_smask
        elif self.span_pool:
            poolout = self.spans_pool(hidden_states,summary_slable)
            del summary_slable
        else:
            poolout = self.pool_ly(hidden_states,summary_smask)
            del summary_smask
        out = self.HEAD( poolout )
        del poolout

        return out

