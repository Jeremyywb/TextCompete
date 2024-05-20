# %run LGBinference.py 
# =========================================================
# Load file
# =========================================================
import gc
import re
from collections import Counter
from nltk.corpus import words
from nltk.corpus import gutenberg
from nltk.corpus import brown
import string
from nltk import FreqDist
from tqdm import tqdm
import pandas as pd
import numpy as np
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy
import yaml 
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
import tokenizers

# =================================================================================
# prepare
stop_words = set(stopwords.words('english'))
tqdm.pandas()
def load_files(args):
    summary_df = pd.read_csv(f'{args.DPath}summaries_test.csv')
    prompt_df =  pd.read_csv(f'{args.DPath}prompts_test.csv')
    return summary_df, prompt_df
def load_train_files(args):
    summary_df = pd.read_csv(f'{args.DPath}summaries_train.csv')
    prompt_df =  pd.read_csv(f'{args.DPath}prompts_train.csv')
    return summary_df, prompt_df
def text_len(text):
    texts = text.split("\n")
    return sum([ len(te.split()) for te in  texts])
modelROOT = '/home/ModelRoot'
cfg_path = f'{modelROOT}/TextCompete/CFG/deberta-v3-base_addPMT.yaml'
with open(cfg_path, 'r') as f:
    args = yaml.safe_load(f)

args = SimpleNamespace(**args)
args.DPath = '/kaggle/input/commonlit-evaluate-student-summaries/'
args.modelRootPath = '/kaggle/input/model-qpmt-rdms'


summary_test_df, prompt_test_df = load_files(args)
summary_train_df, prompt_train_df = load_train_files(args)
corpus1 = brown.words()
# 鍔犺浇绗簩涓鏂欏簱
from nltk.corpus import gutenberg
corpus2 = gutenberg.words()

# 鍚堝苟涓や釜璇枡搴�
combined_corpus = corpus1 + corpus2

# 璁＄畻鍚堝苟鍚庤鏂欏簱鐨勫崟璇嶉鐜�
WORDS = FreqDist(corpus1)

prompt_texts = " ".join(prompt_test_df['prompt_text'].values.tolist()+prompt_train_df['prompt_text'].values.tolist())
prompt_words = nltk.word_tokenize(prompt_texts.lower())
localwordfreq = FreqDist(prompt_words )
InCoupusNum = np.mean([WORDS[c]/localwordfreq[c]  for c in prompt_words if c in WORDS])
for i,wd in enumerate(prompt_words):
    updatefreq = int(localwordfreq[wd]*InCoupusNum*10)
    if i<10:
        print(f"source word:{wd} freq :{localwordfreq[wd]} to {updatefreq}")
    WORDS[wd] = updatefreq
TOT = sum(WORDS.values())


# =====================================================================
# spell correction
# =====================================================================

def words(text): return re.findall(r'\w+', text.lower())

def P(word): 
    "Probability of `word` based on WORDS dictionary."
    # 鐩存帴浠� WORDS 瀛楀吀涓幏鍙栨鐜囧€硷紝濡傛灉鍗曡瘝涓嶅瓨鍦紝榛樿涓� 0.0
    return WORDS.get(word, 0.0)/TOT

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
word_pattern = re.compile(r'\b[A-Za-z]+\b')
def correct_texttokens(texttokens):
    return ' '.join([correction(part) if word_pattern.match(part) else part for part in texttokens])

        
def spelling(word_tokens):
    return list(map(WORDS.get, word_tokens)).count(None)
# =================================================================================

# =====================================================================
# feature func
# =====================================================================

def word_overlap_count(row):
    def check_is_stop_word(word):
        return word in stop_words
    
    prompt_words = row['prompt_tokens']
    summary_words = row['summary_tokens']
    
    if stop_words:
        prompt_words = list(filter(check_is_stop_word, prompt_words))
        summary_words = list(filter(check_is_stop_word, summary_words))
    output = len(set(prompt_words).intersection(set(summary_words)))
    del prompt_words,summary_words
    return output

def ngram_co_occurrence(row, n):
    original_tokens = row['prompt_tokens']
    summary_tokens = row['summary_tokens']
    original_ngrams = set(ngrams(original_tokens, n))
    summary_ngrams = set(ngrams(summary_tokens, n))
    
    common_ngrams = original_ngrams.intersection(summary_ngrams)
    del original_tokens,summary_tokens,original_ngrams,summary_ngrams
    return len(common_ngrams)
def quotes_count(row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        
        if len(quotes_from_summary)>0:
            output = [quote in text for quote in quotes_from_summary].count(True)
        else:
            output = 0
        del summary,text,quotes_from_summary
        return output
def ngrams(token, n):
    ngrams = zip(*[token[i:] for i in range(n)])
    return [" ".join(ngram) for ngram in ngrams]

def pos_count(sent):
    nn_count = 0   #Noun
    pr_count = 0   #Pronoun
    vb_count = 0   #Verb
    jj_count = 0   #Adjective
    uh_count = 0   #Interjection
    cd_count = 0   #Numerics
    
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)

    for token in sent:
        if token[1] in ['NN','NNP','NNS']:
            nn_count += 1

        if token[1] in ['PRP','PRP$']:
            pr_count += 1

        if token[1] in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            vb_count += 1

        if token[1] in ['JJ','JJR','JJS']:
            jj_count += 1

        if token[1] in ['UH']:
            uh_count += 1

        if token[1] in ['CD']:
            cd_count += 1
    
    return pd.Series([nn_count, pr_count, vb_count, jj_count, uh_count, cd_count])


def text_preprocess(prompt_df, summaries):
    prompt_df["prompt_tokens"] = prompt_df["prompt_text"].progress_apply(lambda x: nltk.word_tokenize(x))
    prompt_df["prompt_length"] = prompt_df["prompt_tokens"].map(len)
    summaries["summary_tokens"] = summaries["text"].progress_apply(lambda x:  nltk.word_tokenize(x))
    summaries["summary_length"] = summaries["summary_tokens"].map(len)
    
    #summaries["fixed_summary_text"] = summaries["text"].progress_apply(lambda x: speller(x))
    summaries["splling_err_num"] = summaries["text"].progress_apply(spelling)
    
    df = summaries.merge(prompt_df, how="left", on="prompt_id")
    df['length_ratio'] = df['summary_length'] / df['prompt_length']
    df['word_overlap_count'] = df.progress_apply(word_overlap_count, axis=1)
    df['bigram_overlap_count'] = df.progress_apply(ngram_co_occurrence, args=(2,), axis=1)
    df['bigram_overlap_ratio'] = df['bigram_overlap_count'] / (df['summary_length'] - 1)
    df['trigram_overlap_count'] = df.progress_apply(ngram_co_occurrence, args=(3,), axis=1)
    df['trigram_overlap_ratio'] = df['trigram_overlap_count'] / (df['summary_length'] - 2)
    df['quotes_count'] = df.progress_apply(quotes_count, axis=1)
    return df.drop(columns=["summary_tokens", "prompt_tokens","prompt_length"])



nlp = spacy.load("en_core_web_sm")
# doc = nlp(summary_df['text'].values[4])
# for sent in doc.sents:
#     print()
#     print(sent)
def get_sentlen(text):
    doc = nlp(text)
    return [len(sent.text.split()) for sent in doc.sents]
def sent_feats( text ):
    sentsL = get_sentlen(text)
    return pd.Series([np.mean(sentsL), np.max(sentsL), np.std(sentsL),len(sentsL)])

from textblob import TextBlob

def get_featsX(data):
    data["num_words"] = data["text"].progress_apply(lambda x: len(str(x).split()))
    data["num_unique_words"] = data["text"].progress_apply(lambda x: len(set(str(x).split())))
    data["num_chars"] = data["text"].progress_apply(lambda x: len(str(x)))
    data["num_stopwords"] = data["text"].progress_apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))
    data["num_punctuations"] =data['text'].progress_apply(lambda x: len([c for c in str(x) if c in list(string.punctuation)]))
    data["num_words_upper"] = data["text"].progress_apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    data["num_words_title"] = data["text"].progress_apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    data["mean_word_len"] = data["text"].progress_apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    data["num_paragraphs"] = data["text"].progress_apply(lambda x: len(x.split('\n')))
    data["polarity"] = data['text'].progress_apply(lambda x: TextBlob(x).sentiment[0])
    data["subjectivity"] = data['text'].progress_apply(lambda x: TextBlob(x).sentiment[1])
    return data



from sklearn.metrics.pairwise import cosine_similarity


def _score_paragraphs( paragraphs, summaries,stype=1):
#     paragraphs = [p for p in paragraphs if len(p)>15]
    vectorizer = TfidfVectorizer()
    paragraph_vectors = vectorizer.fit_transform(paragraphs)
    summary_vectors = vectorizer.transform(summaries)
    # 璁＄畻鎬荤粨鍜屾钀戒箣闂寸殑浣欏鸡鐩镐技搴�
    similarities = cosine_similarity(summary_vectors, paragraph_vectors)
    del vectorizer,paragraph_vectors,summary_vectors
    if stype==1:
        return list((np.array(similarities).max(1)-np.array(similarities).min(1))*np.array(similarities).mean(1))
    if stype==2:
        return list( np.array(similarities).max(1))
    if stype==3:
        return list( np.array(similarities).mean(1))
    

# =====================================================================
# exact feature  2
# =====================================================================
def getsimilar(summary_data,prompt_data ):
    stypename = {1:'maxmean',2:'max',3:'mean'}
    folds = list(prompt_data.prompt_id.values)
    # train simi feats
    for stype in [1,2,3]:
        feaname = f'simlitype_{stype}_{stypename[stype]}'
        summary_data[feaname] = 0
        for fold in folds:
            summrs = summary_data[summary_data.prompt_id==fold]['text'].values
            prtext = prompt_data[prompt_data.prompt_id==folds[0]]['prompt_text'].values[0]
            if '\r\n' in prtext :
                promps = prtext.split('\r\n' )
            else:
                promps = prtext.split('\n' )
            scores = _score_paragraphs( promps, summrs,stype)
            summary_data[summary_data.prompt_id==fold][feaname] = scores
            del summrs,prtext,promps
    return summary_data
# ==============================================================================   

# ==============================================================================
# test simi feats
# for stype in [1,2,3]:
#     feaname = f'simlitype_{stype}_{stypename[stype]}'
#     test[feaname] = 0
#     for fold in foldstest:
#         prtext = prompt_df[prompt_df.prompt_id==fold]['prompt_text'].values[0]
#         summrs = test[test.prompt_id==fold]['text'].values
#         if '\r\n' in prtext :
#             promps = prtext.split('\r\n' )
#         else:
#             promps = prtext.split('\n' )
#         scores = _score_paragraphs( promps, summrs,stype)
#         test[test.prompt_id==fold][feaname] = scores
# ==============================================================================     

# =====================================================================
# exact embeddings
# =====================================================================  
from transformers import AutoModel,AutoTokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state.detach().cpu()
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    output = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    del token_embeddings,input_mask_expanded
    return output




class EmbedDataset(torch.utils.data.Dataset):
    def __init__(self,df,tokenizer):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.create_samples()
    def __len__(self):
        return len(self.df)
    def __getitem__(self,idx):
        return self.samples[idx]
    def create_samples(self):
        self.samples = [ ]
        for text in self.df["text"].values:
            tokens = self.tokenizer(
                text,
                None,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=MAX_LEN,return_tensors="pt")
            tokens = {k:v.squeeze(0) for k,v in tokens.items()}
            self.samples.append( tokens )
        del tokens
    

def _append(_this, _new):
    if _this is None: 
        _this = _new
    else:
        _this = np.append(_this, _new, axis=0)
    return _this


MAX_LEN = 640
def get_embeddings(test=None,MODEL_NM='', MAX_LEN=640, BATCH_SIZE=32, verbose=True):
    DEVICE="cuda"
    cpumodel = AutoModel.from_pretrained( MODEL_NM )
    tokenizer = AutoTokenizer.from_pretrained( MODEL_NM )
    dset = EmbedDataset(test, tokenizer)
    dataloader = torch.utils.data.DataLoader(dset,\
                            batch_size=BATCH_SIZE,\
                            shuffle=False)
    model = cpumodel.to(DEVICE)
    del cpumodel
    gc.collect()
    model.eval()
    features = None
    for batch in tqdm(dataloader,total=len(dataloader)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        with torch.no_grad():
            model_output = model(input_ids=input_ids,attention_mask=attention_mask)
        sentence_embeddings = mean_pooling(model_output, attention_mask.detach().cpu())
        # Normalize the embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        sentence_embeddings =  sentence_embeddings.squeeze(0).detach().cpu().numpy()
        if len(sentence_embeddings.shape)==1:
            sentence_embeddings = sentence_embeddings.reshape(1,-1)
        features = _append(features, sentence_embeddings)
        del batch,input_ids,attention_mask,sentence_embeddings,model_output
        torch.cuda.empty_cache()
    if verbose:
        print('Train embeddings shape',features.shape)
    del model,tokenizer;gc.collect()
    del dset,dataloader;gc.collect()
    torch.cuda.empty_cache()
    return features


# =====================================================================
# exact feature 
# =====================================================================

# test = text_preprocess(prompt_test_df, summary_test_df)
train = text_preprocess(prompt_train_df, summary_train_df)
# test = get_featsX(test)
train = get_featsX(train)
# test = getsimilar(test,prompt_test_df)
train = getsimilar(train,prompt_train_df)
# test[['summ_text_len_mean','summ_text_len_max','summ_text_len_std','num_sentences']] = test['text'].progress_apply(sent_feats)
train[['summ_text_len_mean','summ_text_len_max','summ_text_len_std','num_sentences']] = train['text'].progress_apply(sent_feats)
# test[['nn_count','pr_count','vb_count','jj_count','uh_count','cd_count']] = test['text'].progress_apply(pos_count)
train[['nn_count','pr_count','vb_count','jj_count','uh_count','cd_count']] = train['text'].progress_apply(pos_count)

# splling_err_num
# num_punctuations
# ===================================================================================================
# get corrected
# import nltk
# !mkdir nltk_data
# nltk.download('punkt','/kaggle/working/nltk_data')  # 下载分词工具
# nltk.download('stopwords','/kaggle/working/nltk_data') # 下载停用词列表
# nltk.download('brown','/kaggle/working/nltk_data')
# nltk.download('gutenberg','/kaggle/working/nltk_data')
# nltk.download('averaged_perceptron_tagger','/kaggle/working/nltk_data')
# nltk.download("wordnet",'/kaggle/working/nltk_data')
# !unzip /kaggle/working/nltk_data/corpora/wordnet.zip -d /kaggle/working/nltk_data/corpora/
from nltk.stem import WordNetLemmatizer
py_lemmatizer = WordNetLemmatizer()
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.stem.lancaster import LancasterStemmer
lc = LancasterStemmer()
from nltk.stem import SnowballStemmer
sb = SnowballStemmer("english")
from nltk.corpus import words
from nltk.corpus import gutenberg
from nltk.corpus import brown
from nltk import FreqDist
from nltk.corpus import stopwords
import nltk
import re
# nltk.download('wordnet')
from transformers import AutoConfig, AutoModel,AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
pretrainVocab = {k.replace("▁",""):v for k,v in tokenizer.get_vocab().items() }


# =================================================================================
# prepare
stop_words = set(stopwords.words('english'))
tqdm.pandas()

corpus1 = brown.words()
# 加载第二个语料库
from nltk.corpus import gutenberg
corpus2 = gutenberg.words()

# 合并两个语料库
combined_corpus = corpus1 + corpus2

# 计算合并后语料库的单词频率
WORDS = FreqDist(corpus1)
prompt_df = prompt_train_df.copy()

prompt_texts = " ".join(prompt_df['prompt_text'].values.tolist())

punctSplits = [
    '—','-','¨','"','"','(',')','/',   # Added a closing parenthesis
 ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
while "'" in punctSplits:
    punctSplits.remove("'")
def resetText(text):
    for d in punctSplits:
        text =  f' {d} '.join([e for e in text.split(d) if len(e.strip())>0])
    return text
prompt_texts = resetText( prompt_texts )

prompt_words = nltk.word_tokenize(prompt_texts.lower())
localwordfreq = FreqDist(prompt_words )
InCoupusNum = np.mean([WORDS[c]/localwordfreq[c]  for c in prompt_words if c in WORDS])
for i,wd in enumerate(prompt_words):
    updatefreq = int(localwordfreq[wd]*InCoupusNum*10)
    if i<10:
        print(f"source word:{wd} freq :{localwordfreq[wd]} to {updatefreq}")
    WORDS[wd] = updatefreq
TOT = sum(WORDS.values())


# =====================================================================
# spell correction
# =====================================================================

def words(text): return re.findall(r'\w+', text.lower())

def P(word):
    "Probability of `word` based on WORDS dictionary."
    # 直接从 WORDS 字典中获取概率值，如果单词不存在，默认为 0.0
    return WORDS.get(word, 0.0)/TOT

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
word_pattern = re.compile(r'\b[A-Za-z]+\b')
def correct_texttokens(texttokens):
    return ' '.join([correction(part) if word_pattern.match(part) else part for part in texttokens])


punctSplits = [
    '—','-','¨','"','"','(',')','/',   # Added a closing parenthesis
 ',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
while "'" in punctSplits:
    punctSplits.remove("'")
def resetText(text):
    for d in punctSplits:
        text =  f' {d} '.join([e for e in text.split(d) if len(e.strip())>0])
    return text

train['text'] = train['text'].map( resetText )
text_list = train['text'].values

# [c for k,c in correction_dict.items() if c =='egypitian']

import spacy
from spacy.lang.en import English
# punctSplits = [
#     '—','-','¨','"','"','\(','\)'   # Added a closing parenthesis
# ]
# English.Defaults.infixes += punctSplits

nlp = spacy.load('en_core_web_lg', disable=['parser','ner','tagger'])
nlp.vocab.add_flag(lambda s: s.lower() in spacy.lang.en.stop_words.STOP_WORDS, spacy.attrs.IS_STOP)


# for simbols in punctSplits:
#   nlp = addSpliter(simbols,nlp)


# correction('egypitian')## 待处理
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£',
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…',
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─',
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞',
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
puncts10time = "".join(sum([[c]*10 for c in puncts],[]))



import re
from collections import Counter

def viterbi_segment(text):
    probs, lasts = [1.0], [0]
    for i in range(1, len(text) + 1):
        prob_k, k = max((probs[j] * word_prob(text[j:i]), j)
                        for j in range(max(0, i - max_word_length), i))
        probs.append(prob_k)
        lasts.append(k)
    words = []
    i = len(text)
    while 0 < i:
        words.append(text[lasts[i]:i])
        i = lasts[i]
    words.reverse()
    if len(words)>2:
        return [text]
    else:
        return words
#     return words, probs[-1]

def word_prob(word): return dictionary[word] / total
def words(text): return re.findall('[a-z]+', text.lower())
# dictionary = Counter(words(open('big.txt').read()))
dictionary = WORDS.copy()
max_word_length = max(map(len, dictionary))
total = float(sum(dictionary.values()))


docs = nlp.pipe(text_list)




contracs = {
"3/4":"3 quarter",
"They're":"They are",
'3rd':"third",
"2nd":"second",
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldńt": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"weren't":"were not",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"weren´t":"'were not'"
}
misspell_mapping = {
    "likebale":"likeable",
    "withnthe": "with the",
    "societyfollowed":"society followed",
    "Frist":"First",
    'studentdesigned': 'student designed',
    'teacherdesigned': 'teacher designed',
    'genericname': 'generic name',
    'winnertakeall': 'winner take all',
    'studentname': 'student name',
    'driveless': 'driverless',
    'teachername': 'teacher name',
    'propername': 'proper name',
    'bestlaid': 'best laid',
    'genericschool': 'generic school',
    'schoolname': 'school name',
    'winnertakesall': 'winner take all',
    'elctoral': 'electoral',
    'eletoral': 'electoral',
    'genericcity': 'generic city',
    'elctors': 'electoral',
    'venuse': 'venue',
    'blimplike': 'blimp like',
    'selfdriving': 'self driving',
    'electorals': 'electoral',
    'nearrecord': 'near record',
    'egyptianstyle': 'egyptian style',
    'oddnumbered': 'odd numbered',
    'carintensive': 'car intensive',
    'elecoral': 'electoral',
    'oction': 'auction',
    'electroal': 'electoral',
    'evennumbered': 'even numbered',
    'mesalandforms': 'mesa landforms',
    'electoralvote': 'electoral vote',
    'relativename': 'relative name',
    '22euro': 'twenty two euro',
    'ellectoral': 'electoral',
    'thirtyplus': 'thirty plus',
    'collegewon': 'college won',
    'hisher': 'higher',
    'teacherbased': 'teacher based',
    'computeranimated': 'computer animated',
    'canadidate': 'candidate',
    'studentbased': 'student based',
    'gorethanks': 'gore thanks',
    'clouddraped': 'cloud draped',
    'edgarsnyder': 'edgar snyder',
    'emotionrecognition': 'emotion recognition',
    'landfrom': 'land form',
    'fivedays': 'five days',
    'electoal': 'electoral',
    'lanform': 'land form',
    'electral': 'electoral',
    'presidentbut': 'president but',
    'teacherassigned': 'teacher assigned',
    'beacuas': 'because',
    'positionestimating': 'position estimating',
    'selfeducation': 'self education',
    'diverless': 'driverless',
    'computerdriven': 'computer driven',
    'outofcontrol': 'out of control',
    'faultthe': 'fault the',
    'unfairoutdated': 'unfair outdated',
    'aviods': 'avoid',
    'momdad': 'mom dad',
    'statesbig': 'states big',
    'presidentswing': 'president swing',
    'inconclusion': 'in conclusion',
    'handsonlearning': 'hands on learning',
    'electroral': 'electoral',
    'carowner': 'car owner',
    'elecotral': 'electoral',
    'studentassigned': 'student assigned',
    'collegefive': 'college five',
    'presidant': 'president',
    'unfairoutdatedand': 'unfair outdated and',
    'nixonjimmy': 'nixon jimmy',
    'canadates': 'candidate',
    'tabletennis': 'table tennis',
    'himher': 'him her',
    'studentsummerpacketdesigners': 'student summer packet designers',
    'studentdesign': 'student designed',
    'limting': 'limiting',
    'electrol': 'electoral',
    'campaignto': 'campaign to',
    'presendent': 'president',
    'thezebra': 'the zebra',
    'landformation': 'land formation',
    'eyetoeye': 'eye to eye',
    'selfreliance': 'self reliance',
    'studentdriven': 'student driven',
    'winnertake': 'winner take',
    'alliens': 'aliens',
    '2000but': '2000 but',
    'electionto': 'election to',
    'candidatesas': 'candidates as',
    'electers': 'electoral',
    'winnertakes': 'winner takes',
    'isfeet': 'is feet',
    'incar': 'incur',
    'wellconstructed': 'well constructed',
    'craftsmenwomen': 'crafts men women',
    'freelunch': 'free lunch',
    'twothousandrevolutions': 'two thousand revolutions',
    'ushistoryorg': 'us history org',
    'pharohs': 'pharaohs',
    'whitehot': 'white hot',
    'vizers': 'visors',
    'mrjones': 'mr jones',
    'aminute': 'a minute',
    'spoiledmeat': 'spoiled meat',
    'farmersgave': 'farmers gave',
    'spolied': 'spoiled',
    'tradgey': 'tragedy',
    'pyrimid': 'pyramid',
    'pyrimad': 'pyramid',
    'egyptiansfrom': 'egyptians from',
    'harvestthats': 'harvest that',
    'expierment': 'experiment',
    'jestthat': 'jest that',
    'twothousandrevolutionsaminute': 'two thousand revolutions a minute',
    'expirament': 'experiment',
    'nonspoiled': 'non spoiled',
    'egyptains': 'egyptians',
    'tragedys': 'tragedy',
    'pyrmaid': 'pyramid',
    'expirment': 'experiment',
    'whiteit': 'grade there',
    'gradethere': 'tragedy',
    'goverement': 'government',
    'godsthe': 'gods the',
    'paraoh': 'pharaoh',
    'classesupper': 'classes upper',
    'pharoes': 'pharaohs',
    'noblespriests': 'noble priests',
    'farmersslaves': 'farmers slaves',
    'harvestâ€”thatâ€™s': 'harvest that',
    'tradedy': 'tragedy',
    'paraohs': 'pharaohs',
    'paragrapgh': 'paragraph',
    'expieriment': 'experiment',
    'tragdey': 'tragedy',
    'pyramaid': 'pyramid',
    "pyridmaid":"pyramid",
    'pyrmid': 'pyramid',
    'prists': 'priests',
    'pharoas': 'pharaohs',
    'priets': 'priests',
    'pharoph': 'pharaohs',
    'pharaoah': 'pharaohs',
    'pharahos': 'pharaohs',
    'pharaohthe': 'pharaohs',
    "diddernt":"different",
    "vizerier":"vizier",
    "vizire":"vizier",
    "egyptuans":"egyptians",
    "supereriorty":"superiority",
    "ecpirament":" experiment",
    "trajedegy":"tragedy",
}

splling_err_num = []
num_punctuations = []


word_sequences = []
correction_dict = {}
uniwordCover = {}
lemma_dict = {}
unknown_token = set()
new_unkown = []
numnew_unkown_prin = 10
num_wordsCover = 0
num_constra = 0
num_wordlower = 0
num_wordupper = 0
num_wordcapitalize = 0
num_wordstem = 0
num_wordlcstem = 0
num_wordsbstem = 0
num_correction = 0
num_uni_correction = 0
num_correction_invocab = 0
num_lemmaed = 0
num_lemm_cword = 0
num_misspell_mapping = 0
num_words = 0
for doc in tqdm(docs):
    num_punct = 0
    num_err = 0

    word_seq = []
    num_words+=len(doc)
    for token in doc:
        word = token.text.strip()
        if word not in uniwordCover:
            uniwordCover[word] = True
        if token.pos_ is "PUNCT":
            word_seq.append(word)
            num_wordsCover+=1
            num_punct+=1
            continue
        if word in puncts10time:
            word_seq.append(word)
            num_wordsCover+=1
            num_punct+=1
            continue

        if word in pretrainVocab:
            word_seq.append(word)
            num_wordsCover+=1
            continue
        uniwordCover[word] = False

        if word.lower() in contracs:
            word = contracs[word.lower()]
            word_seq.append(word)
            num_constra+=1
            num_err
            continue

        if word.lower() in misspell_mapping:
            word = misspell_mapping[word.lower()]
            word_seq.append(word)
            num_misspell_mapping+=1
            num_err+=1
            continue


        word = token.text.strip().lower()
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordlower+=1
            continue

        word = token.text.strip().upper()
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordupper+=1
            continue

        word = token.text.strip().capitalize()
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordcapitalize+=1
            continue

        word = ps.stem(token.text.strip())
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordstem+=1
            continue

        word = lc.stem(token.text.strip())
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordlcstem+=1
            continue

        word = sb.stem(token.text.strip().strip())
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordsbstem+=1
            continue

        word = token.text.strip()
        if word in lemma_dict:
            lemma = lemma_dict[word]
        else:
            lemma = token.lemma_
        if lemma in pretrainVocab:
            word_seq.append(lemma)
            num_lemmaed +=1
            num_err+=1
            continue

        if len(word)==1:
            unknown_token.add(word)
            word_seq.append(word)
            num_err+=1
            continue
        word = word.lower()
        if len(word)>1:
            if word in correction_dict:
                cword = correction_dict[word]
            else:
                cword = correction(word)
                correction_dict[word] = cword
                num_uni_correction+=1
            num_correction+=1
            if cword in pretrainVocab:
                num_correction_invocab+=1
                word_seq.append(cword)
                num_err+=1
                continue
        lemm_cword = py_lemmatizer.lemmatize(cword)
        if lemm_cword in pretrainVocab:
            word_seq.append(lemm_cword)
            num_lemm_cword +=1
            continue
        splitwords = viterbi_segment(word)
        for word in splitwords:
            if word in pretrainVocab:
                word_seq.append(word)
            else:
                word_seq.append(word)
                if word not in unknown_token:
                    numnew_unkown_prin +=1
                    new_unkown.append(word)
                    if numnew_unkown_prin>=10:
                        print("FIND UNkowns: ",new_unkown)
                        new_unkown=[]
                        numnew_unkown_prin=0
                unknown_token.add(word)
        num_err+=1
    word_sequences.append(word_seq)
    num_punctuations.append( num_punct )
    splling_err_num.append(num_err)


# =============================================================================
# =============================================================================

BATCH_SIZE = 32
dset = EmbedDataset(train)
dataloader = torch.utils.data.DataLoader(dset,\
                        batch_size=BATCH_SIZE,\
                        shuffle=False)
MODEL_NM = "microsoft/deberta-v3-base"
# '/content/gdrive/MyDrive/tmpdata/deberta-v3-large'
features = get_embeddings(dataloader = dataloader,MODEL_NM=MODEL_NM)

BATCH_SIZE = 32

# print("PROSCESSING EMBEDDING")
# MODEL_NM = '../input/deberta-v3-large/deberta-v3-large'
# features = get_embeddings(test,MODEL_NM, MAX_LEN, BATCH_SIZE)
# print("PROSCESSING EMBEDDING DONE")
train['num_punctuations'] = num_punctuations
train['splling_err_num'] = splling_err_num
train['text'] = [' '.join(te) for te in word_sequences]
train[[f'EMBED_{i}' for i in range(features.shape[1])  ]] = features
# test[[f'EMBED_{i}' for i in range(features.shape[1])  ]] = features
# =====================================================================
# parameters and fold
# =====================================================================
# train.to_csv('/content/gdrive/MyDrive/DataUpload/train_features.csv',index=False)

# ======================================================================================
# clean text



# summary_train_df['text_clean'] = summary_train_df['text'].progress_apply(clean_text)
# summary_train_df['text_clean'] = summary_train_df['text_clean'].progress_apply(clean_number)
# summary_train_df['text_clean'] = summary_train_df['text_clean'].progress_apply(clean_misspell)

# test['text_clean'] = test['text'].progress_apply(clean_text)
# test['text_clean'] = test['text_clean'].progress_apply(clean_number)
# test['text_clean'] = test['text_clean'].progress_apply(clean_misspell)
# test.head()
# # ======================================================================================
# summary_train_df["summary_tokens"] = summary_train_df["text_clean"].progress_apply(lambda x:  nltk.word_tokenize(x))
# test["summary_tokens"] = test["text_clean"].progress_apply(lambda x:  nltk.word_tokenize(x))
# summary_train_df["text_clean"] = summary_train_df["summary_tokens"].progress_apply(correct_texttokens)
# test["text_clean"] = test["summary_tokens"].progress_apply(correct_texttokens)


# test = test.groupby('prompt_id').apply(lambda x: x.sort_values('summary_length', ascending=True)).reset_index(drop=True)
# del test['tokenize_length']


# test.to_csv('test_features.csv',index=False)
# summary_train_df.to_csv('cleaned_cleaned.csv')
# 妯″瀷瑕佽缁冩墠鐭ラ亾鏁堟灉
# test
# text_clean


# del test
# del summary_test_df
# del prompt_test_df,summary_train_df,prompt_train_df,corpus1,corpus2,stop_words
# del combined_corpus,WORDS,prompt_texts,prompt_words,localwordfreq
# del InCoupusNum,features,nlp

import gc
gc.collect()


don’t:do not
e’en:(poetic) even
e’er:(poetic) ever
everybody’s:everybody has
everyone’s:everyone has
everything's:everything has
girl's:girl has
guy's:guy has
hadn’t:had not
had’ve:had have
hasn’t:has not
haven’t:have not
he’d:he had
he'll:he shall
he’s:he has
here’s:here is
how’ll:how will
how’re:how are
how’s:how has
I’d:I had
I’d’ve:I would have
I’d’nt:I would not
I’d’nt’ve:I would not have
I’ll:I shall
I’m:I am
I’ve:I have
isn’t:is not
it’d:it would
it’ll:it shall
it’s:it has
let’s:let us
ma’am:(formal) madam
mayn’t:may not
may’ve:may have
mightn’t:might not
might’ve:might have
mine’s:mine is
mustn’t:must not
mustn’t’ve:must not have
must’ve:must have
needn’t:need not
o’clock:of the clock
o’er:over
ol’:old
ought’ve:ought have
oughtn’t:ought not
oughtn’t’ve:ought not have
’round:around
’s:is, has, does, us
shalln’t:shall not (archaic)
shan’:shall not
shan’t:shall not
she’d:she had
she’ll:she shall
she’s:she has
should’ve:should have
shouldn’t:should not
somebody’s:somebody has
someone’s:someone has
something’s:something has
that’ll:that shall
that’s:that has
that’d:that would
there’d:there had
there’ll:there shall
there’re:there are
there’s:there has
these’re:these are
these’ve:these have
they’d:they had
they’d've:they would have
they’ll:they shall
they’re:they are
they’ve:they have
this’s:this has
w’all:we all (Irish
w’at:we at
wanna:want to
wasn’t:was not
we’d:we had
we’d’ve:we would have
we’ll:we shall
we’re:we are
we’ve:we have
weren’t:were not
whatcha:what are you (whatcha doing?)
what:about you (as in asking how someone is today, used as a greeting)
what’d:what did
what’ll:what shall
what’re:what are
what’s:what has
what’ve:what have
when’s:when has
where’d:where did
where’ll:where shall
where’re:where are
where’s:where has
where’ve:where have
which’d:which had
which’ll:which shall
which’re:which are
which’s:which has
which’ve:which have
who’d:who would
who’d’ve:who would have
who’ll:who shall
who’re:who are
who’s:who has
who’ve:who have
why’d:why did
why’re:why are
why’s:why has
willn’t:will not (archaic)
won’t:will not
wonnot:will not (archaic)
would’ve:would have
wouldn’t:would not
wouldn’t’ve:would not have
y’ain’t:you are not
y’all:you all (colloquial
y’all’d’ve:you all would have (colloquial
y’all’d’n’t’ve:you all would not have (colloquial
y’all’re:you all are (colloquial
y’all’ren’t:you all are not (colloquial
yes’m:yes ma’am
yever:have you ever ... ?
y’know:you know
yessir:yes sir
you’d:you had
you’ll:you shall
you’re:you are
you’ve:you have
when’d:when did

