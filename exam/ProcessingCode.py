import pandas as pd
import time
import re


# ========================
# Help Func
# ========================

def remove_options(text):
    pattern = re.compile(r'\nOptions:\n- No\n- Yes')
    cleaned_text = pattern.sub('', text)
    pattern = re.compile(r'\nOptions:\n- Yes\n- No')
    cleaned_text = pattern.sub('', cleaned_text)
    return cleaned_text


def split_A(text):
    # 使用 '\n\n' 分割文本成多个片段
    # segments = text.split('\n\n')
    pattern = re.compile(r'(yes|no)\s*\n\n', re.IGNORECASE)
    segments = pattern.split(text)
    valid_segments = []

    buff = []
    for segment in segments:
      buff.append(segment)
      if ( len(buff)==2 and
        segment.lower() in {'yes', 'no'}):
        valid_segments.append(buff )
        buff = [ ]
      if ( len(buff)==2 and
        segment.lower() not in {'yes', 'no'}):
        buff = [ ]
    return valid_segments

def find_puncidx_before_does(text):
    pattern = re.compile(r'([\n:"])(?=\s*Does\s)')#, re.IGNORECASE
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        return None
def process_text(text):
  L_HQ_A = split_A(
      remove_options(text)
  )
  res = []
  for HQ,A in L_HQ_A:
    idx = find_puncidx_before_does(HQ)
    if idx is None:
      res.append( ("[HQA]",HQ,A) )
    else:
      res.append( (HQ[:idx+1],HQ[idx+1:].replace("\n",""),A) )
  return res


def findHQA(vsl):
 return any([True if v[0]=='[HQA]' else False for v in  vsl])




# ========================
# Load
# ========================

text_df = pd.read_json("/content/sample_data/test.json")

# ========================
# Process
# ========================

start_time = time.time()
text_df['resList2'] = text_df['input'].map(process_text )
text_df['processERR2'] = text_df['resList2'].map( findHQA )


# ========================
# Print
# ========================
print(
    "解析失败句子对个数",
    sum([len(x) for x in text_df[text_df['processERR2']==True]['resList2'].values])
)

print(
    '解析成功句子对个数:',
    sum([len(x) for x in text_df[text_df['processERR2']==False]['resList2'].values])
)


elapsed_time = time.time() - start_time
print("运行时间: {:.6f} 秒".format(elapsed_time))


# ========================
# Saving
# ========================

jsondata = []
for id1,vls in zip(text_df['id'].values,text_df['resList2'].values):
  for i,vl in enumerate(vls):
    idv = f'{id1}-{i}'
    jsondata.append(
        {"id": idv,
        "head":vl[0],
        "Question": vl[1],
        "Answer": vl[2]
  })

import json
with open('/content/sample_data/AdaptLLM-finance-tasks-Headline.json', 'w') as f:
    json.dump(jsondata, f)