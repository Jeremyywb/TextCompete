import lightgbm as lgb
import numpy as np

# gbm = lgb.Booster(model_file='model.txt')

# LgbPara ={
        
#         'objective': 'regression_l1',
#         'metric': 'mae',
#         'learning_rate': 0.02,
#         'num_leaves': 36,
#         'lambda_l1': 0.3,
#         'lambda_l2': 10,
#         'max_depth': 10,
#         'seed': 33,
#         'feature_fraction': 0.5,
#         'bagging_fraction': 0.9,
#         'bagging_freq': 1,
#         'feature_fraction_seed': 7,
# #         'min_data_in_leaf': 20,
#         'num_threads':8,
#         'verbose': -1
#     }


def get_fold(args, fold):
    train_folds = [c for c in args.selected_folds if c != fold]
    train_feats = None
    tarin_target = None
    for fd in train_folds:
        file_path = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{fd}_best_LGBFEAT.npz"
        loaded_data = np.load(file_path)
        feature = loaded_data["feature"]
        target = loaded_data["target"]
        del loaded_data
        if train_feats is None: 
            train_feats = feature
        else:
            train_feats = np.append(train_feats, feature, axis=0)
        if tarin_target is None: 
            tarin_target = target
        else:
            tarin_target = np.append(tarin_target, target, axis=0)
        del feature,target
    valid_file = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{fold}_best_LGBFEAT.npz"
    loaded_data = np.load(valid_file)
    val_feats = loaded_data["feature"]
    val_target = loaded_data["target"]
    del loaded_data
    return train_feats,tarin_target,val_feats,val_target


def lgb_train(args,fold,LgbPara):
    lgb_model_prefix = f"{args.modelRootPath}/{args.name}_{args.save_name_prefix}__fold{args.fold}_"
    x_train,y_train,x_valid,y_valid = get_fold(args, fold)
    oof_len = len(x_valid)
    oof = np.zeros((oof_len,2))
    for i, targetN in enumerate(args.model['target_cols']):
        dtrain = lgb.Dataset(x_train, label=y_train[:,i])
        dvalid = lgb.Dataset(x_valid, label=y_valid[:,i]) 
        gbm = lgb.train(
            params=LgbPara,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            early_stopping_rounds=50,
            feval = smspe_lgb,
            verbose_eval=200
        )
        valpred = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
        oof[:,i] = valpred
        gbm.save_model(f'{lgb_model_prefix}_{targetN}_lgbmodel.txt')
    return oof,y_valid
