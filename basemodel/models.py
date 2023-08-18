import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
import torch.utils.checkpoint
import torch.nn.functional as F
from TextCompete.basemodel.pooling import NLPPooling
from W_Informer.models.domian_encoder import PromptAwareEncoder,ConvPoolLayer
from torch.cuda.amp import autocast


def top_n_layer_freeze(module,n):
    for i in range(0,n,1):
        for n,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False


class CommonLitModelV1(nn.Module):
    def __init__(
        self,
        CrosAttPara,
        CrosConvPara,
        CrosenEoderPara,
        model_name,
        span_pool,
        gradient_checkpointing,
        multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        init_head,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True) if not config_path else torch.load(config_path)
        self.config.update(
                                
                        {
                            "hidden_dropout":0.0,
                            "hidden_dropout_prob": 0.0,
                            "attention_dropout":0.0,
                            "attention_probs_dropout_prob": 0.0,
                        }
                            )
        self.backbone = AutoModel.from_pretrained(model_name,config=self.config) if not config_path else AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
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
        self.encoders = PromptAwareEncoder(**CrosenEoderPara)
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

        # self.norm = nn.LayerNorm(self.config.hidden_size)
        # self.norm2= nn.LayerNorm(self.config.hidden_size)
        if multilfc:
            outprojs = [nn.Linear(finaldim, finaldim//2),
                        # nn.BatchNorm1d(finaldim//2),
                        nn.ELU(),
                        nn.Linear(finaldim//2, 64),
                        nn.ELU(),
                        nn.Linear(64, 2),
                         ]
        else:
            outprojs = [nn.Linear(finaldim, 2) ]


        #==================
        # if add relu
        # result*1.08-2
        #==================


        if self.activation=='relu6':
            outprojs.append(nn.ReLU6())
        elif self.activation=='leakyrelu':
            outprojs.append(nn.LeakyReLU(0.2))
        

        self.outproj = nn.Sequential(*outprojs)

        if self.init_head:
            self.outproj.apply(self._init_weights)

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        
    def _init_conv_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')
            
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

    def feature(
        self,
        inputs,
        prompt_inputs
                     ):
        hidden_states = self.backbone(inputs['input_ids'], inputs['attention_mask'])[0]
        # print(hidden_states.shape)

        if self.add_prompt:
            p_hidden_states = self.backbone(prompt_inputs['input_ids'], prompt_inputs['attention_mask'])[0]
            hidden_states = self.encoders(
                    hidden_states,
                    p_hidden_states,
                    prompt_inputs['attention_mask']
                )
            del p_hidden_states

        if self.multilpool:
            out = torch.cat([
                self.pool_ly(hidden_states,inputs['smask']),
                self.spans_pool(hidden_states,inputs['slable']),
                 ],dim=-1)
        elif self.span_pool:
            out = self.spans_pool(hidden_states,inputs['slable'])
        else:
            out = self.pool_ly(hidden_states,inputs['smask'])
        del hidden_states
        return out

    @autocast()
    def forward(
        self, 
        inputs,
        prompt_inputs
                    ):
        feature = self.feature(inputs, prompt_inputs)
        if self.activation == 'relu6':
            feature = self.outproj(feature)*1.08-2
        elif self.activation == 'sigmoid':
            feature = self.outproj(feature)*3.25+1.25
        else:
            feature = self.outproj(feature)
        return feature

