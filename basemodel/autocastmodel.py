import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
import torch.utils.checkpoint
from torch.cuda.amp import autocast
import torch.nn.functional as F
from TextCompete.basemodel.pooling import NLPPooling
from W_Informer.models.domian_encoder import PromptAwareEncoder,ConvPoolLayer
from TextCompete.basemodel.autocastheads import *
from TextCompete.basemodel.models import (
    CommonLitModelV1,CommonLitModelV2,CommonLitModelV3,CommonLitModelV4,CommonLitModelV5
    )
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


# ============================================
# model V2
class AUTOCommonLitModelV2(CommonLitModelV2):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCommonLitModelV2, self).__init__(*args, **kwargs)
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
            # _Lens = torch.tensor([0]+ self.split_text_length(hidden_states.shape[1])).to(prompt_input_ids.device)
            p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]
            scores = self.adaPoll2D3D(hidden_states, p_hidden_states,summary_smask)
            # scores = torch.einsum(
            #     "ble,bse->bls", hidden_states, p_hidden_states
            #     )
            # scores = self.encoders( 
            #     scores.unsqueeze(1) ).view( p_hidden_states.shape[0],-1 )

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
        del hidden_states
        if self.add_prompt:
            out = self.HEAD( torch.cat([poolout,scores]),dim=-1 )
        else:
            out = self.HEAD( poolout)
        del poolout

        return out


# ============================================
# model V2
class AUTOCommonLitModelV3(CommonLitModelV3):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCommonLitModelV3, self).__init__(*args, **kwargs)
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
        prompt_embedding=None,
        summ_sentence_embedding=None
                            ):

        hidden_states = self.backbone(
            input_ids=summary_input_ids, 
            attention_mask = summary_attention_mask,
            token_type_ids = summary_token_type_ids 
            )[0]

        del summary_input_ids,summary_attention_mask,summary_token_type_ids
        # if self.add_prompt:
        #     # _Lens = torch.tensor([0]+ self.split_text_length(hidden_states.shape[1])).to(prompt_input_ids.device)
        #     # p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]

        #     hidden_states = self.encoders(
        #             hidden_states,
        #             prompt_embedding,
        #             # prompt_inputs['attention_mask']
        #         )
        #     # hidden_states = torch.cat([ self.encoders(
        #     #         hidden_states[:,_Lens[idx]:_Lens[idx+1],:],
        #     #         prompt_embedding,
        #     #         # prompt_inputs['attention_mask']
        #     #     ) for idx in range(len(_Lens)-1) ],dim=1)

        #     del prompt_embedding

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
        del hidden_states
        if self.add_prompt:
            poolout = torch.cat(
                [ poolout,summ_sentence_embedding,
                prompt_embedding, torch.abs(summ_sentence_embedding-prompt_embedding)]
                ,dim=-1
            )
            del summ_sentence_embedding,prompt_embedding
        out = self.HEAD( poolout )
        del poolout

        return out




class AUTOCommonLitModelV4(CommonLitModelV4):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCommonLitModelV4, self).__init__(*args, **kwargs)
        self.HEAD = AUTOHEAD_MAPPER[self.headname]( 
            finaldim   = self.finaldim,
            output_dim = self.output_dim,
            init_head  = self.init_head,
            config     = self.config  
        )

    @autocast()
    def forward(
        self, 
        summary_sentence_idx,
        summary_input_ids,
        summary_attention_mask,
        prompt_sentence_idx,
        prompt_input_ids,
        prompt_attention_mask,

                            ):
        cumidx = 0 
        Shidden_states = None
        for idx in summary_sentence_idx.squeeze(0):
            cumidx+=idx
            hidden_states = self.backbone(
                input_ids=summary_input_ids[:,cumidx-idx:cumidx ] , 
                attention_mask = summary_attention_mask[:,cumidx-idx:cumidx ],
                )[0]
            if Shidden_states is None:
                Shidden_states = hidden_states
            else:
                Shidden_states = torch.cat([Shidden_states, hidden_states],dim=1)
        Spoolout = self.pool_ly(Shidden_states, summary_attention_mask)
        del Shidden_states,hidden_states,summary_sentence_idx,summary_attention_mask

        cumidx = 0 
        Shidden_states = None
        # print(prompt_attention_mask.shape)
        # print(prompt_attention_mask)
        for idx in prompt_sentence_idx.squeeze(0):
            cumidx+=idx
            # print(cumidx)
            hidden_states = self.backbone(
                input_ids=prompt_input_ids[:,cumidx-idx:cumidx ] , 
                attention_mask = prompt_attention_mask[:,cumidx-idx:cumidx ],
                )[0]
            if Shidden_states is None:
                Shidden_states = hidden_states
            else:
                Shidden_states = torch.cat([Shidden_states, hidden_states],dim=1)
        Ppoolout = self.pool_ly(Shidden_states, prompt_attention_mask)
        del Shidden_states,hidden_states,prompt_sentence_idx,prompt_attention_mask
        output = torch.cat([Ppoolout,Spoolout, torch.abs(Spoolout - Ppoolout )], dim=-1)
        del Spoolout , Ppoolout
        out = self.HEAD( output )
        del output
        return out



class AUTOCommonLitModelV5(CommonLitModelV5):
    def __init__(
        self,
        # use_mdn,
        *args, **kwargs):
        super(AUTOCommonLitModelV5, self).__init__(*args, **kwargs)
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
            # _Lens = torch.tensor([0]+ self.split_text_length(hidden_states.shape[1])).to(prompt_input_ids.device)
            p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]

            # hidden_states = self.encoders(
            #         hidden_states,
            #         p_hidden_states,
            #         # prompt_inputs['attention_mask']
            #     )
            # hidden_states = torch.cat([ self.encoders(
            #         hidden_states[:,_Lens[idx]:_Lens[idx+1],:],
            #         p_hidden_states,
            #         # prompt_inputs['attention_mask']
            #     ) for idx in range(len(_Lens)-1) ],dim=1)
            
            prompt_poolout = self.pool_ly(p_hidden_states,prompt_attention_mask)
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
        del hidden_states
        poolout = torch.cat([prompt_poolout,poolout,torch.abs(poolout-prompt_poolout)],dim=-1)
        out = self.HEAD( poolout )
        del poolout,prompt_poolout

        return out

