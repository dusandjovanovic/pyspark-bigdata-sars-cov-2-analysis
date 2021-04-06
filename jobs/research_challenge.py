import numpy as np
import pandas as pd
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import os
import re

meta = pd.read_csv("../data/open-research/metadata.csv")

meta = meta[((meta['pdf_json_files'] == True) | (meta['pmc_json_files'] == True))]
meta_sm = meta[['cord_uid', 'sha', 'pmcid', 'title', 'abstract', 'publish_time', 'url']]
meta_sm.drop_duplicates(subset="title", keep=False, inplace=True)
meta_sm.loc[meta_sm.publish_time == '2020-12-31'] = "2020-03-31"
meta_sm.head()

sys.path.insert(0, "../")

root_path = '../data/open-research/'
df = {"paper_id": [], "text_body": []}
df = pd.DataFrame.from_dict(df)

collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

for i, file_name in enumerate(collect_json):
    row = {"paper_id": None, "text_body": None}
    if i % 2000 == 0:
        print("====processed " + str(i) + ' json files=====')
        print()

    with open(file_name) as json_data:

        data = json.load(json_data, object_pairs_hook=OrderedDict)

        row['paper_id'] = data['paper_id']

        body_list = []

        for _ in range(len(data['body_text'])):
            try:
                body_list.append(data['body_text'][_]['text'])
            except:
                pass

        body = "\n ".join(body_list)

        row['text_body'] = body
        df = df.append(row, ignore_index=True)

merge1 = pd.merge(meta_sm, df, left_on='sha', right_on=['paper_id'])
merge2 = pd.merge(meta_sm, df, left_on='pmcid', right_on=['paper_id'])
merge2.head()

merge_final = merge2.append(merge1, ignore_index=True)
merge_final.drop_duplicates(subset="title", keep=False, inplace=True)

merge_final = merge2.append(merge1, ignore_index=True)
merge_final.drop_duplicates(subset="title", keep=False, inplace=True)

corona = merge_final[(merge_final['publish_time'] > '2019-11-01') & (
    merge_final['text_body'].str.contains('nCoV|Cov|COVID|covid|SARS-CoV-2|sars-cov-2'))]
corona.shape

def clean_dataset(text):
    text = re.sub('[\[].*?[\]]', '', str(text))  # remove in-text citation
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)  # remove hyperlink
    text = re.sub(r'\\b[A-Z a-z 0-9._ - ]*[@](.*?)[.]{1,3} \\b', '', text)  # remove email
    text = re.sub(r'^a1111111111 a1111111111 a1111111111 a1111111111 a1111111111.*[\r\n]*', ' ',
                  text)  # have no idea what is a11111.. is, but I remove it now
    text = re.sub(r'  +', ' ', text)  # remove extra space
    text = re.sub('[,\.!?]', '', text)
    text = re.sub(r's/ ( *)/\1/g', '', text)
    text = re.sub(r'[^\w\s]', '', text)  # strip punctions (recheck)
    return text


corona['text_body'] = corona['text_body'].apply(clean_dataset)
corona['title'] = corona['title'].apply(clean_dataset)
corona['abstract'] = corona['abstract'].apply(clean_dataset)
corona['text_body'] = corona['text_body'].map(lambda x: x.lower())
coro = corona.reset_index(drop=True)
coro.head()

coro['count_abstract'] = coro['abstract'].str.split().map(len)
coro['count_abstract'].sort_values(ascending=True)

y = np.array(coro['count_abstract'])
sns.distplot(y);

coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)

import seaborn as sns
import matplotlib.pyplot as plt

y = np.array(coro['count_abstract'])

sns.distplot(y);

coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)

y = np.array(coro['count_text'])
sns.distplot(y);

coro2=coro[((coro['count_text']>500)&(coro['count_text']<4000))]
coro2.to_csv("corona.csv",index=False)
test=coro2[coro2['count_abstract']<5]
train= coro2.drop(test.index)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)