
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
        from torch.cuda.amp import autocast
        def autocast_forward(cls):
            class NewClass(cls):
                def __init__(self, *args, **kwargs):
                    super(NewClass, self).__init__(*args, **kwargs)
                @autocast()
                def forward(self, *args, **kwargs):
                    return super(NewClass, self).forward(*args, **kwargs)
            
            return NewClass

        headname = args.model['params']['headname']
        CommonLitModelV1Train = autocast_forward(CommonLitModelV1)
        CommonLitModelV1Train.HEADCLASS = autocast_forward(HEAD_MAPPER[headname])


        model_parameters.update({"pretrained":True,
                              "config_path":None })#影响model中 dropout配置
        model =  CommonLitModelV1Train(**model_parameters)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    return tokenizer, model

def download_configs(args):
    tokenizer = AutoTokenizer.from_pretrained(args.download)
    config = AutoConfig.from_pretrained(args.download, output_hidden_states=True) 
    tokenizer.save_pretrained(args.tokenizer_path)
    torch.save(config, args.config_path)
    del tokenizer,config