# =============================
#  Model
# =============================


import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel,AutoTokenizer
import torch.utils.checkpoint
from torch.cuda.amp import autocast
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
}

def top_n_layer_freeze(module,n):
    # for _name,p in module.embeddings.named_parameters():
    #     p.requires_grad = False
    for i in range(0,n,1):
        for _name,p in module.encoder.layer[i].named_parameters():
            p.requires_grad = False


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
        headname,
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
        self.HEAD = HEAD_MAPPER[headname]( finaldim,output_dim, init_head, self.config )

        if freezing>0:
            top_n_layer_freeze(self.backbone,freezing)
        # self.encoders.apply(self._kaimin)
        if REINIT_LAYERS>0:
            for layer in self.backbone.encoder.layer[-REINIT_LAYERS:]:
                # for module in layer.modules():
                    # self._xavier_init(module)
                layer.apply(self._init_weights)

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
        del hidden_states
        out = self.HEAD( poolout )
        del poolout

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

    if (not os.path.exists(args.config_path)
        and args.do_train):
        download_configs(args)

    model_parameters = {}
    model_parameters.update( args.model['params'] )
    # model_parameters.update(
    #     {"config_path":args.config_path,
    #     "download":args.download}
    #     )
    model_parameters.update( {"download":args.download}  )

    _update = ['CrosConvPara','CrosenEoderPara','pooling_params','spans_pooling_params','CrosAttPara']
    for _name in _update:
        model_parameters[_name] = args.model[_name]
    if args.do_inference:
        model_parameters.update( {"pretrained":False,
                              "config_path":args.config_path } )
        model = CommonLitModelV1(**model_parameters)
        state = torch.load(args.foldModel,
            map_location=torch.device('cpu')
                )
        model.load_state_dict(state)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if args.do_train:
        model_parameters.update({"pretrained":True,
                              "config_path":None })#影响model中 dropout配置
        model =  CommonLitModelV1(**model_parameters)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    return tokenizer, model

def download_configs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.download)
    config = AutoConfig.from_pretrained(args.download, output_hidden_states=True) 
    tokenizer.save_pretrained(args.tokenizer_path)
    torch.save(config, args.config_path)
    del tokenizer,config

# ==================================================================================



