# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 23:20:03 2021

@author: Pablo González
"""

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt


def read_data():
    true_news = pd.read_csv(r'data/True.csv')
    fake_news = pd.read_csv(r'data/Fake.csv')
    # Fake news receive a 1, while real news receive a 0
    true_news["label"] = np.zeros(len(true_news), dtype=int)
    fake_news["label"] = np.ones(len(fake_news), dtype=int)
    output = pd.concat((true_news,fake_news),axis = 0 )
    # We will consider both the main content as well as the title (both lowercase) 
    # as the textual information provided by the news.
    output.text = output.title.str.lower() + ' ' + output.text.str.lower()
    
    return output.drop('title', axis = "columns")

def tokenize(vec):
    # Roberta seems like a really good approach:
    # https://medium.com/analytics-vidhya/evolving-with-bert-introduction-to-roberta-5174ec0e7c82
    # https://huggingface.co/transformers/model_doc/roberta.html
    # https://pytorch.org/hub/pytorch_fairseq_roberta/
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation = True)
    return [tokenizer(i, max_length = 966) for i in vec]


def plot_count(df):
    index = sorted(pd.to_datetime(df.date, errors='coerce').unique())
    index = pd.date_range(index[0], index[-1])
    df2 = pd.DataFrame( [], index = index )
    df2['fake'] = pd.to_datetime(df.query('label == 1').date, errors='coerce').value_counts().sort_index()
    df2['real'] = pd.to_datetime(df.query('label == 0').date, errors='coerce').value_counts().sort_index()
    df2 = df2.fillna(0)
    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(df2.fake, color = 'red', linewidth=1)
    plt.plot(df2.real, color = 'blue', linewidth=1)
    plt.title( 'Amount of fake and real news per day' )
    plt.grid(alpha = 0.8)
    plt.xlabel("Date")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.legend(['Fake News','Real News'])
    plt.show()

    
# functions for doc2vec
def get_tagged_documents(df, document_column, tags_column):
    '''tokenise and tag documents with id, required for doc2vec input'''
    
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from tqdm.notebook import tqdm
    from joblib import Parallel, delayed
    from gensim.utils import tokenize

    df_dask = dd.from_pandas(df, npartitions=120)
    
    def get_tagged_doc(row):
        '''tokenise and tag a row in a dataframe'''
        return TaggedDocument(words=list(tokenize(row[document_column], lowercase=True)), tags=row[tags_column])
    
    with ProgressBar():
        df_tagged = df_dask.map_partitions(lambda df: df.apply(lambda row: get_tagged_doc(row), axis=1)).compute(scheduler='processes')
        
    return df_tagged

def generate_doc2vec_model(df_tagged, method="dm", max_epochs=10, vec_size=50, alpha=0.025, min_count=50):
    '''build doc2vec model provided tagged documents'''
    
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from tqdm.notebook import tqdm
    from joblib import Parallel, delayed
    from gensim.utils import tokenize

    # check whether to use DM, or DBOW
    if method=="dm":
        dm_val = 1
    else:
        dm_val = 0
    
    # build doc2vec model
    model = Doc2Vec(
        df_tagged.values
        ,window=100
        ,alpha=alpha
        ,min_count=min_count
        ,dm=dm_val
        ,epochs=max_epochs
        ,workers=8
    )
    
    return model

def get_vector_representations(model, docs):
    '''get vector representations for a column of text'''
    
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from tqdm.notebook import tqdm
    from joblib import Parallel, delayed
    from gensim.utils import tokenize

    def infer_vector(model, doc):
        # note, 20 steps means 20 iterations to update the randomly initialised weights
        return model.infer_vector(list(tokenize(doc, lowercase=True)), steps=20)
    
    vector_representation = Parallel(n_jobs=8, prefer="threads")(delayed(infer_vector)(model, doc) for doc in tqdm(docs))
    
    return pd.DataFrame.from_records(vector_representation)



def autok(seq, tokenizer, max = 100):
  ids = []
  masks = []
  for i in seq:
    temp = tokenizer.encode_plus( i, truncation=True, return_tensors="pt", max_length= max,
                                padding="max_length", return_attention_mask=True)
    ids.append(temp['input_ids'])
    masks.append(temp['attention_mask'])
  output = torch.stack( [torch.cat( ids, dim = 0), torch.cat( masks, dim = 0)] , dim = 0)
  output = output.transpose(1,0)
  return output
