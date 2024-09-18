# =============================
#  Model
# =============================


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
import torch.utils.checkpoint
from tqdm import tqdm
import torch.nn.functional as F
from TextCompete.basemodel.pooling import NLPPooling
from W_Informer.models.domian_encoder import PromptAwareEncoder,ConvPoolLayer
from TextCompete.basemodel.heads import *
import TextCompete.metrics_loss.loss_function as mdn
import os

HEAD_MAPPER = {
    "conv1d_01":CONVHEAD1,
    'dense_01':DENSEHEAD1,
    'uniform_conv1d_01':UniFormCONVHEAD1,
    'uniform_dense_01':UniFormDENSEHEAD1,
    'binssoftmax_01':BINSSOFTMAX,
    'bndense_01':BNDENSEHEAD1,
    'binssoftmax2conv1d':BINSOFTMAX2CONV1D,
}

def top_n_layer_freeze(module,n):
    # for _name,p in module.embeddings.named_parameters():
    #     p.requires_grad = False
    for i in range(0,n,1):
        for _name,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False

# ====================================================
# model V1
class CommonLitModelV1(nn.Module):
    def __init__(
        self,
        # use_mdn,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        span_pool,
        gradient_checkpointing,
        # multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        output_dim,
        pretrained,
        download,
        headname=None,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        # ============================================
        # backbone setting
        # BUGGGG : AutoModel.from_config init randomly
        # ============================================
        if config_path is None:
            self.config = AutoConfig.from_pretrained(download, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        # self.config = torch.load(config_path)
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(download, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ============================================
        # init para
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt

        d_model = self.config.hidden_size
        CrosAttPara['d_model'] = CrosConvPara['d_model'] = CrosenEoderPara['d_model'] =CrosConvPara['c_in'] = d_model
        CrosAttPara['d_ff'] = CrosAttPara['d_model']*4
        CrosenEoderPara['attParameter'].update(CrosAttPara)
        CrosenEoderPara['downConvPara'].update(CrosConvPara)
        # ============================================

        # ============================================
        # heads..prompt encoder
        if self.add_prompt:
            self.encoders = PromptAwareEncoder(**CrosenEoderPara)

        # ============================================
        # heads..pooling
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        if self.span_pool:
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update(
                {"in_features":self.config.hidden_size,
                "out_features":self.config.hidden_size}
            )
            self.spans_pool = NLPPooling(**self.spans_pooling_params)
        if self.multilpool:
            finaldim = self.spans_pooling_params['out_features'] + self.pooling_params['out_features']
        elif self.span_pool:
            finaldim = self.spans_pooling_params['out_features']
        else:
            finaldim = self.pooling_params['out_features']

        # ============================================
        # heads..HEAD proj
        self.finaldim = finaldim
        self.headname = headname
        self.output_dim = output_dim
        self.init_head = init_head

        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

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

    def split_text_length(self,L):
        segments = []
        cumL = 0
        while L > 320:
            cumL+=256
            segments.append(cumL)
            L -= 256
        segments.append(L+cumL)
        return segments

    def forward(
        self, 
        summary_input_ids,
        summary_attention_mask,
        summary_token_type_ids=None,
        summary_slable=None,
        summary_smask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None
                            ):

        hidden_states = self.backbone(
            input_ids=summary_input_ids, 
            attention_mask = summary_attention_mask,
            # token_type_ids = summary_token_type_ids 
            )[0]

        del summary_input_ids,summary_attention_mask,summary_token_type_ids
        if self.add_prompt:
            # _Lens = torch.tensor([0]+ self.split_text_length(hidden_states.shape[1])).to(prompt_input_ids.device)
            p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]

            hidden_states = self.encoders(
                    hidden_states,
                    p_hidden_states,
                    # prompt_inputs['attention_mask']
                )
            # hidden_states = torch.cat([ self.encoders(
            #         hidden_states[:,_Lens[idx]:_Lens[idx+1],:],
            #         p_hidden_states,
            #         # prompt_inputs['attention_mask']
            #     ) for idx in range(len(_Lens)-1) ],dim=1)

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
        out = self.HEAD( poolout )
        del poolout

        return out



# ====================================================
# model V2
class CommonLitModelV2(nn.Module):
    def __init__(
        self,
        # use_mdn,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        span_pool,
        gradient_checkpointing,
        # multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        output_dim,
        pretrained,
        download,
        headname=None,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        # ============================================
        # backbone setting
        # BUGGGG : AutoModel.from_config init randomly
        # ============================================
        if config_path is None:
            self.config = AutoConfig.from_pretrained(download, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        # self.config = torch.load(config_path)
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(download, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ============================================
        # init para
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt

        d_model = self.config.hidden_size
        CrosAttPara['d_model'] = CrosConvPara['d_model'] = CrosenEoderPara['d_model'] =CrosConvPara['c_in'] = d_model
        CrosAttPara['d_ff'] = CrosAttPara['d_model']*4
        CrosenEoderPara['attParameter'].update(CrosAttPara)
        CrosenEoderPara['downConvPara'].update(CrosConvPara)
        # ============================================

        # ============================================
        # heads..prompt encoder
        if self.add_prompt:
            self.encoders = nn.Sequential(
                nn.Conv2d(1,1,kernel_size=3,stride=1,padding="same"),
                nn.ELU(),
                nn.MaxPool2d(2,stride=2 ),
                nn.Conv2d(1,1,kernel_size=5,stride=1,padding="same"),
                nn.ELU(),
                nn.MaxPool2d(3,stride=3),
                nn.Conv2d(1,1,kernel_size=7,stride=1,padding="same"),
                nn.ELU(),
                nn.AdaptiveMaxPool2d(16)
            )
            self.encoders.apply(self._trunc_norm)
            self.grama2w = nn.MaxPool2d(2,stride=1)
            self.grama3w = nn.MaxPool2d(3,stride=1)
            self.grama2wAda = nn.AdaptiveMaxPool1d(256)
            self.grama3wAda = nn.AdaptiveMaxPool1d(256)
        # ============================================
        # heads..pooling
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        if self.span_pool:
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update(
                {"in_features":self.config.hidden_size,
                "out_features":self.config.hidden_size}
            )
            self.spans_pool = NLPPooling(**self.spans_pooling_params)
        if self.multilpool:
            finaldim = self.spans_pooling_params['out_features'] + self.pooling_params['out_features']
        elif self.span_pool:
            finaldim = self.spans_pooling_params['out_features']
        else:
            finaldim = self.pooling_params['out_features']

        # ============================================
        # heads..HEAD proj
        self.finaldim = finaldim
        self.headname = headname
        self.output_dim = output_dim
        self.init_head = init_head
        if self.add_prompt:
            self.finaldim += 512

        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

    def _trunc_norm(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.2, a=-1.96*0.2, b=1.96*0.2)
            module.bias.data.zero_()

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

    def adaPoll2D3D(self,hidden_states, p_hidden_states,attmask):
        scores = scores = torch.einsum(
                "ble,bse->bls", hidden_states, p_hidden_states
                )
        scale = 1/torch.sqrt(attmask.sum(-1))
        scale = scale.unsqueeze(1).unsqueeze(2).expand(scores.size())
        scores = torch.softmax(scale*scores,dim=-1)
        attmask_expanded = attmask.unsqueeze(-1).expand(scores.size()).float()
        scores = scores*attmask_expanded
        output_2gram = self.grama2w(scores.unsqueeze(1)).squeeze(1)
        output_3gram = self.grama3w(scores.unsqueeze(1)).squeeze(1)
        del scores
        output_2gram = self.grama2wAda(torch.max(output_2gram,dim=1).values)
        output_2gram = self.grama3wAda(torch.max(output_3gram,dim=1).values)
        return torch.cat([output_2gram,output_3gram],dim=-1)#(bs,512)


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


class CommonLitModelV3(nn.Module):
    def __init__(
        self,
        # use_mdn,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        span_pool,
        gradient_checkpointing,
        # multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        output_dim,
        pretrained,
        download,
        headname=None,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        # ============================================
        # backbone setting
        # BUGGGG : AutoModel.from_config init randomly
        # ============================================
        if config_path is None:
            self.config = AutoConfig.from_pretrained(download, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        # self.config = torch.load(config_path)
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(download, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ============================================
        # init para
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt

        d_model = self.config.hidden_size
        CrosAttPara['d_model'] = CrosConvPara['d_model'] = CrosenEoderPara['d_model'] =CrosConvPara['c_in'] = d_model
        CrosAttPara['d_ff'] = CrosAttPara['d_model']*4
        CrosenEoderPara['attParameter'].update(CrosAttPara)
        CrosenEoderPara['downConvPara'].update(CrosConvPara)
        # ============================================

        # ============================================
        # heads..prompt encoder
        if self.add_prompt:
            self.encoders = PromptAwareEncoder(**CrosenEoderPara)

        # ============================================
        # heads..pooling
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        if self.span_pool:
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update(
                {"in_features":self.config.hidden_size,
                "out_features":self.config.hidden_size}
            )
            self.spans_pool = NLPPooling(**self.spans_pooling_params)
        if self.multilpool:
            finaldim = self.spans_pooling_params['out_features'] + self.pooling_params['out_features']
        elif self.span_pool:
            finaldim = self.spans_pooling_params['out_features']
        else:
            finaldim = self.pooling_params['out_features']

        # ============================================
        # heads..HEAD proj
        self.finaldim = finaldim*2
        self.headname = headname
        self.output_dim = output_dim
        self.init_head = init_head

        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

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

    def split_text_length(self,L):
        segments = []
        cumL = 0
        while L > 320:
            cumL+=256
            segments.append(cumL)
            L -= 256
        segments.append(L+cumL)
        return segments

    def forward(
        self, 
        summary_input_ids,
        summary_attention_mask,
        summary_token_type_ids,
        summary_slable=None,
        summary_smask=None,
        text_emb_distance=None,
        # summ_sentence_embedding=None
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
                [ poolout, 
                  text_emb_distance,
                 # torch.abs(summ_sentence_embedding-prompt_embedding)
                ]
                ,dim=-1
            )
            del summ_sentence_embedding,prompt_embedding
        out = self.HEAD( poolout )
        del poolout

        return out





class CommonLitModelV4(nn.Module):
    def __init__(
        self,
        # use_mdn,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        span_pool,
        gradient_checkpointing,
        # multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        output_dim,
        pretrained,
        download,
        headname=None,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        # ============================================
        # backbone setting
        # BUGGGG : AutoModel.from_config init randomly
        # ============================================
        if config_path is None:
            self.config = AutoConfig.from_pretrained(download, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        # self.config = torch.load(config_path)
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(download, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)

        # ============================================
        # init para
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt

        d_model = self.config.hidden_size
        CrosAttPara['d_model'] = CrosConvPara['d_model'] = CrosenEoderPara['d_model'] =CrosConvPara['c_in'] = d_model
        CrosAttPara['d_ff'] = CrosAttPara['d_model']*4
        CrosenEoderPara['attParameter'].update(CrosAttPara)
        CrosenEoderPara['downConvPara'].update(CrosConvPara)
        # ============================================

        # ============================================
        # heads..prompt encoder
        if self.add_prompt:
            self.encoders = PromptAwareEncoder(**CrosenEoderPara)

        # ============================================
        # heads..pooling
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        if self.span_pool:
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update(
                {"in_features":self.config.hidden_size,
                "out_features":self.config.hidden_size}
            )
            self.spans_pool = NLPPooling(**self.spans_pooling_params)
        if self.multilpool:
            finaldim = self.spans_pooling_params['out_features'] + self.pooling_params['out_features']
        elif self.span_pool:
            finaldim = self.spans_pooling_params['out_features']
        else:
            finaldim = self.pooling_params['out_features']

        # ============================================
        # heads..HEAD proj
        self.finaldim = finaldim*3
        self.headname = headname
        self.output_dim = output_dim
        self.init_head = init_head

        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

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
        for idx in summary_sentence_idx:
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
        for idx in prompt_sentence_idx:
            cumidx+=idx
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


class CommonLitModelV5(nn.Module):
    def __init__(
        self,
        # use_mdn,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        span_pool,
        gradient_checkpointing,
        # multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        output_dim,
        pretrained,
        download,
        headname=None,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        # ============================================
        # backbone setting
        # BUGGGG : AutoModel.from_config init randomly
        # ============================================
        if config_path is None:
            self.config = AutoConfig.from_pretrained(download, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        # self.config = torch.load(config_path)
        # self.config.hidden_dropout = 0.
        # self.config.hidden_dropout_prob = 0.
        # self.config.attention_dropout = 0.
        # self.config.attention_probs_dropout_prob = 0.
        if pretrained:
            self.backbone = AutoModel.from_pretrained(download, config=self.config)
        else:
            self.backbone = AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # ============================================
        # init para
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt

        d_model = self.config.hidden_size
        CrosAttPara['d_model'] = CrosConvPara['d_model'] = CrosenEoderPara['d_model'] =CrosConvPara['c_in'] = d_model
        CrosAttPara['d_ff'] = CrosAttPara['d_model']*4
        CrosenEoderPara['attParameter'].update(CrosAttPara)
        CrosenEoderPara['downConvPara'].update(CrosConvPara)
        # ============================================

        # ============================================
        # heads..prompt encoder
        if self.add_prompt:
            self.encoders = PromptAwareEncoder(**CrosenEoderPara)

        # ============================================
        # heads..pooling
        self.pooling_params = pooling_params
        self.pooling_params.update({"in_features":self.config.hidden_size,
                                    "out_features":self.config.hidden_size
                                    })
        self.pool_ly = NLPPooling(**self.pooling_params)
        
        if self.span_pool:
            self.spans_pooling_params = spans_pooling_params
            self.spans_pooling_params.update(
                {"in_features":self.config.hidden_size,
                "out_features":self.config.hidden_size}
            )
            self.spans_pool = NLPPooling(**self.spans_pooling_params)
        if self.multilpool:
            finaldim = self.spans_pooling_params['out_features'] + self.pooling_params['out_features']
        elif self.span_pool:
            finaldim = self.spans_pooling_params['out_features']
        else:
            finaldim = self.pooling_params['out_features']

        # ============================================
        # heads..HEAD proj
        self.finaldim = finaldim*2
        self.headname = headname
        self.output_dim = output_dim
        self.init_head = init_head

        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        # if freezing>0:
        #     top_n_layer_freeze(self.backbone,freezing)


        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

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

    def split_text_length(self,L):
        segments = []
        cumL = 0
        while L > 320:
            cumL+=256
            segments.append(cumL)
            L -= 256
        segments.append(L+cumL)
        return segments

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
        poolout = torch.cat([poolout,torch.abs(poolout-prompt_poolout)],dim=-1)
        out = self.HEAD( poolout )
        del poolout,prompt_poolout

        return out


# =============================================================
# sentence transformer


def embed_text(text, tokenizer, model,device):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input.to(device))
    attention_mask = encoded_input['attention_mask']
    token_embedding = model_output[0]
    del encoded_input, model_output
    return token_embedding.cpu(), attention_mask.cpu()

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask



def splitWith(text, punct):
    SPLIT_SIMPLE = "[SPLIT]" 
    texts = text.split(punct)
    # texts = [te.strip() for te in texts if len(te.strip())>0]
    tejoins =punct + SPLIT_SIMPLE
    return tejoins.join(texts).split(SPLIT_SIMPLE)


def chunktext(test):
    stext = []
    for tes in splitWith(test, "."):
        tesQues = splitWith(tes, "?")
        for tesQ in tesQues:
            stext.extend(splitWith(tesQ, "!"))
    stext = [c for c in stext if len(c)>0]
    MAXLEN = 256
    ONEHalfMax = int(MAXLEN*1.5)
    TEXTS = []
    GATHERLEN = 0
    GATHERTEX = None
    for te in stext:
        _len = len(te.split())
        if _len>MAXLEN:
            TEXTS.append(te)
            if GATHERTEX is not None:
                TEXTS.append(GATHERTEX)
                GATHERTEX = None
                GATHERLEN = 0
            continue

        if GATHERTEX is None:
            GATHERTEX = te
            GATHERLEN = _len
        elif GATHERLEN+_len>ONEHalfMax:
            TEXTS.append(GATHERTEX)
            GATHERTEX = te
            GATHERLEN = _len
        else:
            GATHERTEX+=  te
            GATHERLEN += _len

        if GATHERLEN>MAXLEN:
            TEXTS.append(GATHERTEX)
            GATHERTEX = None
            GATHERLEN = 0
    if GATHERTEX is not None:
        TEXTS.append(GATHERTEX)
    return TEXTS

def embedAndPoll_textes( textlist,tokenizer, model,device ):
    token_embeddings, attention_masks = None, None
    for text in textlist:
        token_embedding, attention_mask = embed_text(text, tokenizer, model,device)
        if token_embeddings is None:
            token_embeddings = token_embedding
        else:
            token_embeddings = torch.cat([token_embeddings, token_embedding],dim=1)
        if attention_masks is None:
            attention_masks = attention_mask
        else:
            attention_masks = torch.cat([attention_masks, attention_mask],dim=1)
        del attention_mask, token_embedding
        
    poolout = mean_pooling(token_embeddings, attention_masks)
    del token_embeddings, attention_masks
    return poolout.squeeze(0)

def senembd(text,tokenizer, model,device):
    TEXTS = chunktext(text)
    return embedAndPoll_textes( TEXTS,tokenizer, model,device )


# =============================================================



def make_prompt_embdeeding(summary_df,prompt_df, device, args):
    #============================================
    #if is inference download model bin
    # use pretrained model and config
    # ***do after load_from_pretrained***
    # if not hasattr(args,'pretraibins'):
    #     raise ValueError('args.pretraibins not exists')

    # #============================================
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # config = torch.load(args.config_path)
    # model = AutoModel.from_pretrained(args.pretraibins, config=config)
    # # pool = MeanPoolingA()
    # # pool_ly(hidden_states,summary_smask)
    # prompt_inputs = {}
    # for prompt_id,prompt_text in zip(prompt_df['prompt_id'].values,
    #                                  prompt_df['prompt_text'].values):

    #     inputs = tokenizer.encode_plus(
    #                         prompt_text,
    #                         return_tensors=None,
    #                         add_special_tokens=True,
    #                         max_length = 2048,
    #                         # pad_to_max_length = True,#IF not sort with len,will be True
    #                         truncation = True
    #                 )
    #     for k, v in inputs.items():
    #         inputs[k] = torch.tensor(v, dtype=torch.long).unsqueeze(0)
    #     with torch.no_grad():
    #         embed = model(**inputs)[0]
    #     prompt_inputs = torch.mean( embed,dim=1)


        # # print("embed.shape:",embed.shape)
        # if embed.shape[1]<512:
        #     paddL = 512-embed.shape[1]
        #     paddV = torch.zeros(embed.shape[0],paddL,embed[2],dtype=torch.float)
        #     embed = torch.cat([embed,paddV],dim=1)
        # if embed.shape[1]>512:
        #     embed = torch.cat([embed[:,:511,:],embed[:,-1,:].unsqueeze(1) ],dim=1)
        # prompt_inputs[prompt_id] = embed.squeeze(0)
        # print("prompt_inputs:embed.shape:",embed.shape)
    if not hasattr(args,'SentenceModelName'):
        raise ValueError('args.SentenceModelName not exists')
    SentenceModelName = args.SentenceModelName
    # "microsoft/mpnet-base"
    # modelname = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(SentenceModelName)
    model = AutoModel.from_pretrained(SentenceModelName).to(device)

    textlist = tqdm(summary_df['text'].values)
    summary_sentence_embs = []
    for text in textlist:
        summary_sentence_embs.append(senembd(text,tokenizer, model,device))
    prompt_inputs = {}
    for prompt_id,prompt_text in zip(prompt_df['prompt_id'].values,
                                     prompt_df['prompt_text'].values):
        prompt_inputs[prompt_id] = senembd(prompt_text,tokenizer, model, device)
    del model,tokenizer,textlist
    
    return summary_sentence_embs,prompt_inputs
# ================================================================================





# ============================================================
# load from pretrained tokenizer/model for train and inference
def load_from_pretrained(args,get_tokenizer=True):
    """Load the pretrained model and tokenizer."""

    # In distributed training, the .from_pretrained methods guarantee that only
    # one local process can concurrently perform this procedure.
    if not hasattr(args, "modelRootPath"):
        print("args.modelRootPath Not Set")
        raise ValueError('args Setting Error')
    if not hasattr(args, "do_inference"):
        print("args.do_inference Not Set")
        raise ValueError('args attr Error')
    if not hasattr(args, "do_train"):
        print("args.do_train Not Set")
        raise ValueError('args attr Error')
    if not hasattr(args, "pretraibins"):
        raise ValueError('pretraibins not exists')
    # pretraibins modelName + 'modebins'
    if args.do_train and args.do_inference:
        print("Can't do train and Inference at a time")
        raise ValueError('args do_train and do_inference ValueError')
    if args.trainer['HEADTrain']:
        args.model['params']['freezing'] = 12
    args.config_path = f"{args.modelRootPath}/{args.name}_config.pth"
    args.tokenizer_path = f"{args.modelRootPath}/{args.name}_tokenizer"
    args.foldModel = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{args.fold}_best.pth"
    headname = args.model['params']['headname']
    args.headModel = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{args.fold}_{headname}best.pth"

    if (not os.path.exists(args.config_path)
        and args.do_inference):
        print("Inference model config path not exists")
        raise ValueError("model path Error")

    # if (not os.path.exists(args.config_path)
    #     and args.do_train):
    #     download_configs(args)

    model_parameters = {}
    model_parameters.update( args.model['params'] )
    model_parameters.update( {"download":args.download}  )

    _update = ['CrosConvPara','CrosenEoderPara','pooling_params','spans_pooling_params','CrosAttPara']
    for _name in _update:
        model_parameters[_name] = args.model[_name]
    if args.do_inference:
        model_parameters.update( {"pretrained":False,
                                  "config_path":args.config_path } )
        model = eval(args.version)(**model_parameters)
        state = torch.load(args.foldModel,
            map_location=torch.device('cpu')
                )
        model.load_state_dict(state)
        del state
        if get_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.do_train:
        from TextCompete.basemodel.autocastmodel import (
                AUTOCommonLitModelV2,AUTOCommonLitModelV1,AUTOCommonLitModelV3,AUTOCommonLitModelV4,AUTOCommonLitModelV5
            )

        model_parameters.update({"pretrained":True,
                              "config_path":None })#影响model中 dropout配置
        modelName = f'AUTO{args.version}'
        model =  eval(modelName)(**model_parameters)
        torch.save(model.config, args.config_path)
        if get_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(args.download)
            tokenizer.save_pretrained(args.tokenizer_path)
    if get_tokenizer:
        return tokenizer, model
    else:
        return model

# def download_configs(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.download)
#     config = AutoConfig.from_pretrained(args.download, output_hidden_states=True) 
#     tokenizer.save_pretrained(args.tokenizer_path)

#     # ============================大问题。。。。。
#     torch.save(config, args.config_path)
#     del tokenizer,config

# ==================================================================================

