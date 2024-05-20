contracs = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
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
"you've": "you have"
}


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
num_words = 0
for doc in tqdm(docs):
    word_seq = []
    num_words+=len(doc)
    for token in doc:
        word = token.text.strip()
        if word not in uniwordCover:
            uniwordCover[word] = True
        if token.pos_ is "PUNCT":
            word_seq.append(word)
            num_wordsCover+=1
            continue
        if word in pretrainVocab:
            word_seq.append(word)
            num_wordsCover+=1
            continue
        uniwordCover[word] = False

        if word in contracs:
            word = contracs[word]
            word_seq.append(word)
            num_constra+=1
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
            continue

        if len(word)==1:
            unknown_token.add(word)
            word_seq.append(word)
            continue

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
                word_seq.append(word)
            else:
                word_seq.append('[UNK]')
                if cword not in unknown_token:
                    numnew_unkown_prin +=1
                    new_unkown.append(cword)
                    if numnew_unkown_prin>=10:
                        print("FIND UNkowns: ",new_unkown)
                        new_unkown=[]
                        numnew_unkown_prin=0
                unknown_token.add(cword)
    word_sequences.append(word_seq)
