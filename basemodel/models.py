# =============================
#  Model
# =============================

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
import torch.utils.checkpoint
import torch.nn.functional as F
from TextCompete.basemodel.pooling import NLPPooling
from W_Informer.models.domian_encoder import PromptAwareEncoder,ConvPoolLayer
import os

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
        span_pool,
        gradient_checkpointing,
        multilfc,
        multilpool,
        add_prompt,
        activation,
        freezing,
        REINIT_LAYERS,
        init_head,
        config_path=None,
        pooling_params={},
        spans_pooling_params = {},
                        ):
        super().__init__()

        self.config = torch.load(config_path)
        self.config.hidden_dropout = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.attention_dropout = 0.
        self.config.attention_probs_dropout_prob = 0.
        self.backbone = AutoModel.from_config(self.config)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        self.span_pool = span_pool
        self.multilpool = multilpool
        self.activation = activation
        self.init_head = init_head
        self.add_prompt = add_prompt
        self.multilfc = multilfc

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
            outprojs = [
                
                nn.Conv1d(finaldim, finaldim//2, 1),
                nn.BatchNorm1d(finaldim//2),
                nn.ELU(),
                nn.Dropout(0.2),
                nn.Conv1d(finaldim//2, 64, 1),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Conv1d(64, 2, 1)
                         ]
        else:
            outprojs = [nn.Linear(finaldim, 2) ]


        #==================
        # if add relu
        # result*1.08-2
        #==================

        self.outproj = nn.Sequential(*outprojs)

        if self.init_head:
            self.outproj.apply(self._init_weights)

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

    def _kaimin(self, module):
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight,mode='fan_in',nonlinearity='leaky_relu')

    def _xavier_init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
                
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
        summary_input_ids,
        summary_attention_mask,
        summary_slable=None,
        summary_smask=None,
        prompt_input_ids=None,
        prompt_attention_mask=None
                            ):

        hidden_states = self.backbone(summary_input_ids, summary_attention_mask)[0]
        del summary_input_ids,summary_attention_mask
        if hidden_states.shape[0]==1 and self.multilfc:
            raise ValueError("BATCH SIZE 1 dose not support multilfc here")
        if self.add_prompt:
            p_hidden_states = self.backbone(prompt_input_ids, prompt_attention_mask)[0]
            hidden_states = self.encoders(
                    hidden_states,
                    p_hidden_states,
                    # prompt_inputs['attention_mask']
                )
            del p_hidden_states,prompt_attention_mask,prompt_input_ids

        if self.multilpool:
            out = torch.cat([
                self.pool_ly(hidden_states,summary_smask),
                self.spans_pool(hidden_states,summary_slable),
                 ],dim=-1)
            del summary_slable,summary_smask
        elif self.span_pool:
            out = self.spans_pool(hidden_states,summary_slable)
            del summary_slable
        else:
            out = self.pool_ly(hidden_states,summary_smask)
            del summary_smask
        del hidden_states
        if self.multilfc:
            out = self.outproj(out.unsqueeze(-1)).squeeze(-1)
        else:
            out = self.outproj(out)
        
        return out




# ============================================================
# load from pretrained tokenizer/model for train and inference
def load_from_pretrained(args):
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
    if args.do_train and args.do_inference:
        print("Can't do train and Inference at a time")
        raise ValueError('args do_train and do_inference ValueError')

    args.config_path = f"{args.modelRootPath}/{args.name}_config.pth"
    args.tokenizer_path = f"{args.modelRootPath}/{args.name}_tokenizer"
    args.foldModel = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{args.fold}_best.pth"

    if (not os.path.exists(args.config_path)
        and args.do_inference):
        print("Inference model config path not exists")
        raise ValueError("model path Error")

    if (not os.path.exists(args.config_path)
        and args.do_train):
        download_configs(args)

    model_parameters = {}
    model_parameters.update( args.model['params'] )
    model_parameters.update(
        {"config_path":args.config_path}
        )

    _update = ['CrosConvPara','CrosenEoderPara','pooling_params','spans_pooling_params','CrosAttPara']
    for _name in _update:
        model_parameters[_name] = args.model[_name]
    if args.do_inference:
        model = CommonLitModelV1(**model_parameters)
        state = torch.load(args.foldModel,
        #                            map_location=torch.device('cpu')
                )
        model.load_state_dict(state)
    if args.do_train:
        model =  CommonLitModelV1(**model_parameters)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    return tokenizer, model

def download_configs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.download)
    config = AutoConfig.from_pretrained(args.download, output_hidden_states=True) 
    tokenizer.save_pretrained(args.tokenizer_path)
    torch.save(config, args.config_path)

# ==================================================================================

