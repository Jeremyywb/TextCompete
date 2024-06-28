import pandas as pd
# train = pd.read_csv('/kaggle/input/cess-lgb-feature/train_features.csv')
# test = pd.read_csv('/kaggle/working/test_features.csv')
from sklearn.preprocessing import MinMaxScaler
NoneFea =['student_id', 'prompt_id', 'text', 'content', 'wording','prompt_question', 'prompt_title', 'prompt_text']
text_features = [c for c in train.columns if c not in NoneFea and 'EMBED_' not in c ]
scaler = MinMaxScaler().fit(train[text_features])
train[text_features] = scaler.transform(train[text_features])
test[text_features] = scaler.transform(test[text_features])


def get_fold(data,fold,feats,tarcols):
    x_train = data[data['prompt_id']!=fold][feats].values
    y_train = data[data['prompt_id']!=fold][tarcols].values
    x_valid = data[data['prompt_id']==fold][feats].values
    y_valid = data[data['prompt_id']==fold][tarcols].values
    return x_train,y_train,x_valid,y_valid 

def _append(_this, _new):
    if _this is None: 
        _this = _new
    else:
        _this = np.append(_this, _new, axis=0)
    return _this

LgbPara ={
        
        'objective': 'regression_l1',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 20,
        'lambda_l1': 0.9,
        'lambda_l2': 10,
        'max_depth': 10,
        'seed': 33,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction_seed': 7,
#         'min_data_in_leaf': 20,
        'num_threads':8,
        'verbose': -1
    }




tarcols = ['content', 'wording']
NoneFea =['student_id', 'prompt_id', 'text', 'content', 'wording','prompt_question', 'prompt_title', 'prompt_text']
feats = [c for c in train.columns if c not in NoneFea and 'tfidf' not in c and 'simlitype_' not in c]

x_test = test[feats].values


# =====================================================================
# folds run
# =====================================================================

import lightgbm as lgb
import numpy as np

folds = list(train.prompt_id.unique())
tarcols = ['content', 'wording']
test[tarcols] = 0
oofs = None
refs = None
for fold in folds:
    print(f"train on prompt_id:{fold}")
    x_train,y_train,x_valid,y_valid = get_fold( train,fold,feats,tarcols)
    oof_len = len(x_valid)
    oof = np.zeros((oof_len,2))
    for i, targetN in enumerate(tarcols):
        print(f"train on target:{targetN}")
        dtrain = lgb.Dataset(x_train, label=y_train[:,i])
        dvalid = lgb.Dataset(x_valid, label=y_valid[:,i]) 
        evaluation_results = {}
        gbm = lgb.train(
            params=LgbPara,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
#             early_stopping_rounds=50,
#             verbose_eval=200,
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=True),
                lgb.log_evaluation(100),
                lgb.callback.record_evaluation(evaluation_results)
            ],
        )
        valpred = gbm.predict(x_valid, num_iteration=gbm.best_iteration)
        test[targetN] += gbm.predict(x_test, num_iteration=gbm.best_iteration)/len(folds)
        oof[:,i] = valpred
    oofs = _append(oofs, oof)
    refs = _append(refs, y_valid)


# =====================================================================
# scors
# =====================================================================
from sklearn.metrics import mean_squared_error
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

print(MCRMSE(refs, oofs))


# =====================================================================
# saving
# =====================================================================
submission = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/sample_submission.csv')
submission = submission.drop(columns=tarcols).merge(test[['student_id'] + tarcols], on='student_id', how='left')
print(submission.head())
submission[['student_id'] + tarcols].to_csv('lgb_submission.csv', index=False)


['don’t:do not ',
 'e’en:even',
 'e’er:ever',
 'everybody’s everybody has ',
 'everyone’s  everyone has ',
 "everything's    everything has ",
 "girl's  girl has ",
 "guy's   guy has ",
 'hadn’t  had not',
 'had’ve  had have',
 'hasn’t  has not',
 'haven’t have not',
 'he’d    he had ',
 "he'll   he shall ",
 'he’s    he has ',
 'here’s  here is',
 'how’ll  how will ',
 'how’re  how are',
 'how’s   how has ',
 'I’d I had ',
 'I’d’ve  I would have',
 'I’d’nt  I would not',
 'I’d’nt’ve   I would not have',
 'I’ll    I shall ',
 'I’m I am',
 'I’ve    I have',
 'isn’t   is not',
 'it’d    it would',
 'it’ll   it shall ',
 'it’s    it has ',
 'let’s   let us',
 'ma’am (formal)  madam',
 'mayn’t  may not',
 'may’ve  may have',
 'mightn’t    might not',
 'might’ve    might have',
 'mine’s  mine is',
 'mustn’t must not',
 'mustn’t’ve  must not have',
 'must’ve must have',
 'needn’t need not',
 'o’clock of the clock',
 'o’er    over',
 'ol’ old',
 'ought’ve    ought have',
 'oughtn’t    ought not',
 'oughtn’t’ve ought not have',
 '’round  around',
 '’s  is, has, does, us ',
 'shalln’t    shall not (archaic)',
 'shan’   shall not',
 'shan’t  shall not',
 'she’d   she had ',
 'she’ll  she shall ',
 'she’s   she has ',
 'should’ve   should have',
 'shouldn’t   should not',
 'somebody’s  somebody has ',
 'someone’s   someone has ',
 'something’s something has ',
 'that’ll that shall ',
 'that’s  that has ',
 'that’d  that would ',
 'there’d there had ',
 'there’ll    there shall ',
 'there’re    there are',
 'there’s there has ',
 'these’re    these are',
 'these’ve    these have',
 'they’d  they had ',
 "they’d've   they would have ",
 'they’ll they shall ',
 'they’re they are ',
 'they’ve they have',
 'this’s  this has ',
 'w’all   we all (Irish',
 'w’at    we at',
 'wanna   want to',
 'wasn’t  was not',
 'we’d    we had ',
 'we’d’ve we would have',
 'we’ll   we shall ',
 'we’re   we are',
 'we’ve   we have',
 'weren’t were not',
 'whatcha what are you (whatcha doing?)',
 'what about you (as in asking how someone is today, used as a greeting)',
 '',
 'what’d  what did',
 'what’ll what shall ',
 'what’re what are ',
 'what’s  what has ',
 'what’ve what have',
 'when’s  when has ',
 'where’d where did',
 'where’ll    where shall ',
 'where’re    where are',
 'where’s where has ',
 'where’ve    where have',
 'which’d which had ',
 'which’ll    which shall ',
 'which’re    which are',
 'which’s which has ',
 'which’ve    which have',
 'who’d   who would ',
 'who’d’ve    who would have',
 'who’ll  who shall ',
 'who’re  who are',
 'who’s   who has ',
 'who’ve  who have',
 'why’d   why did',
 'why’re  why are',
 'why’s   why has ',
 'willn’t will not (archaic)',
 'won’t   will not',
 'wonnot  will not (archaic)',
 'would’ve    would have',
 'wouldn’t    would not',
 'wouldn’t’ve would not have',
 'y’ain’t you are not ',
 'y’all   you all (colloquial',
 'y’all’d’ve  you all would have (colloquial',
 'y’all’d’n’t’ve  you all would not have (colloquial',
 'y’all’re    you all are (colloquial',
 'y’all’ren’t you all are not (colloquial',
 'yes’m   yes ma’am ',
 'yever   have you ever ... ?',
 'y’know  you know',
 'yessir  yes sir',
 'you’d   you had ',
 'you’ll  you shall ',
 'you’re  you are',
 'you’ve  you have',
 'when’d  when did']



 