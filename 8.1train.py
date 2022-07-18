import time
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sentence_transformers.util import cos_sim
import sys
import key_words
from IPython.display import clear_output
import pickle

def save_obj(obj:object,name:str):
    ext = '.pickle'
    with open(name + ext, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name:str)->object:
    ext = '.pickle'
    with open(name + ext, 'rb') as handle:
        return pickle.load(handle)

train_path = "data/train_phase1.tsv"
df = pd.read_csv(train_path, sep="\t")

def maximal_marginal_relevance(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words,
        top_n = 5,
        diversity = 0.8):
    """
    Maximal Marginal Relevance algorithm for keyword extraction
    * from KeyBERT repository on github

    Args:
        doc_embedding (numpy.ndarray): embedding of shape (1, 768)
        word_embeddings (numpy.ndarray): embedding of shape (N, 768)
        words (List[str]): list of words
        top_n (Optional[int]): number of top words to extract
        diversity (Optional[float]): diversity of top words to extract

    Returns:
        List[Tuple[str, float]]: list of top_n words with their scores
    """
    # make sure 2d array
    if doc_embedding.ndim == 1:
        doc_embedding = doc_embedding.reshape(1, -1)

    # Extract similarity within words, and between words and the document

    word_doc_similarity = np.array(cos_sim(word_embeddings, doc_embedding)).clip(-1, 1).round(6)
    word_similarity = np.array(cos_sim(word_embeddings, word_embeddings)).clip(-1, 1).round(6)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate maximal_marginal_relevance
        mmr = (1-diversity) * candidate_similarities -\
            diversity * target_similarities.reshape(-1, 1)
        # if return mmr is empty
        if mmr.size == 0:
            continue
        mmr = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr)
        candidates_idx.remove(mmr)

    return [words[idx] for idx in keywords_idx]

def emb_keywords(keywords):
    # x = np.array(list(map(lambda k: np.array(list(map(lambda x: BERT.model.encode(str(x)), k))),keywords)))
    print(len(keywords),keywords)
    x = np.array(list(map(lambda k: np.array(list(map(lambda x: model.encode(str(x)), k))),keywords)))
    if x.ndim == 3:
        x = x.reshape(max(x.shape[1],x.shape[0]),x.shape[2])
    return x

def get_candidates(n_grams, doc):
    x = list(map(lambda gram :
        key_words.candidates_tokens(str(doc), n_gram_range=gram)
        , n_grams))
    return x

def match_keywords(
    keywords_emb:list[np.ndarray],
    candidates_emb : list[np.ndarray],
    thershold: float)\
        -> float:
    """
    match keywords with candidates in a document

    Args:
        keywords_emb (List[np.ndarray]): list of keywords embeddings
        candidates_emb (List[np.ndarray]): list of document's candidates embeddings
        thershold (float): threshold

    Returns:
        float: score
    
    example:
        >>> match_keywords(keywords_emb, candidates_emb, thershold=0.5)
        >>> 0.8
    """
    similarities = list(map(lambda cand:
                    cos_sim(np.array(keywords_emb), cand.reshape(cand.shape[0],cand.shape[1])).__array__().max(axis=1).round(6).clip(-1, 1),
                    candidates_emb))
    return similarities

def keywords_pipeline(docs,batch = 20):
    n_docs = len(docs)
    if n_docs < batch:
        batch = n_docs
    times = []
    docs_keys_ls = []
    n_gram = [(2,3)]
    # all model answers
    t1 = time.perf_counter()
    docs_keys_ls_s = []
    # do in batches
    for i in range(0,n_docs,batch):
        print("Processing batch {}/{}".format(i//batch+1, (n_docs//batch)+1))
        students_candidates = list(map(lambda doc: get_candidates(n_gram, doc), docs[i:i+batch]))
        students_candidates_emb =  list(map( lambda st: list(map( emb_keywords ,st)), students_candidates))
        docs_emb = emb_keywords(docs[i:i+batch])
        docs_keywords = list(map(lambda x: maximal_marginal_relevance(
                x[0].reshape(1, -1),x[1],x[2],top_n=10 ,diversity=0.7),
                zip(docs_emb,students_candidates_emb,students_candidates)))
        docs_keys_ls_s.extend(docs_keywords)

    if n_docs % batch != 0 and n_docs > batch:
        students_candidates = list(map(lambda doc: get_candidates([n_gram], doc), docs[i+batch:]))
        students_candidates_emb =  list(map( lambda st: list(map( emb_keywords ,st)), students_candidates))
        docs_emb = emb_keywords(docs[i+batch:])
        docs_keywords = list(map(lambda x: maximal_marginal_relevance(
                x[0].reshape(1, -1),x[1],x[2],top_n=10 ,diversity=0.7),
                zip(docs_emb,students_candidates_emb,students_candidates)))
        docs_keys_ls_s.extend(docs_keywords)

    docs_keys_ls.append(docs_keys_ls_s)
    times.append(time.perf_counter() - t1)
    print("time",times)
    return docs_keys_ls

def grading(keywords_embeddings_list,students_candidates_emb_list,thershold=0.5):
    """
    Args:
        keywords_embeddings_list: list of list of list of embeddings
        students_candidates_emb_list: list of list of list of embeddings
        thershold: thershold for the similarity
    Returns:
        a list of list of list of grades
    """
    grades = []
    for i in range(len(keywords_embeddings_list)):
        grades.append(np.array(list(map(lambda st_cand:
                match_keywords(keywords_embeddings_list[i], st_cand,
                thershold=thershold),
                students_candidates_emb_list[i]
                ))))
    grades = np.array(list(map(lambda sim: (sim.__array__().max(axis=1) >thershold).sum(axis=1)/float(sim.shape[-1]) , grades)))
    return grades

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

words_emb_dict = {}
def get_word_emb(word):
    if word in words_emb_dict:
        return words_emb_dict[word]
    else:
        words_emb_dict[word] = model.encode(word)
        return words_emb_dict[word]

def get_words_emb(words):
    return list(map(get_words_emb, words))

model_answers_dict = {
    1: load_obj(f'data/essaySet_{1}_model_answers'),
    2: load_obj(f'data/essaySet_{2}_model_answers'),
    3: load_obj(f'data/essaySet_{3}_model_answers'),
    4: load_obj(f'data/essaySet_{4}_model_answers'),
    5: load_obj(f'data/essaySet_{5}_model_answers'),
    6: load_obj(f'data/essaySet_{6}_model_answers'),
    7: load_obj(f'data/essaySet_{7}_model_answers'),
    8: load_obj(f'data/essaySet_{8}_model_answers'),
    9: load_obj(f'data/essaySet_{9}_model_answers'),
    10: load_obj(f'data/essaySet_{10}_model_answers'),
}


times = {}
for essay in range(3,11):
    t = []
    print(f"EssaySet: {essay} ...")
    t1 = time.perf_counter()
    model_answers = model_answers_dict[essay]
    docs = df.query(f'EssaySet == {essay}')["EssayText"].values.tolist()[:10]
    t.append(time.perf_counter() - t1)

    docs_keywords = keywords_pipeline(docs, batch=5)

    print("docs keywords emb ...")
    t1 = time.perf_counter()
    # docs_keywords_emb = list(map(lambda x: get_words_emb(x),docs_keywords))
    docs_keywords_emb = list(map(model.encode,docs_keywords))
    t.append(time.perf_counter() - t1)

    print("model keywords ...")
    t1 = time.perf_counter()
    model_candidates = list(map(lambda ans: candidates_tokens(ans,n_gram_range=(2,3)),model_answers))
    # model_candidates = list(map(lambda ans: key_words.candidates_tokens(ans,n_gram_range=(2,3)),model_answers))
    model_candidate_emb = list(map(lambda cand: get_words_emb(cand),model_candidates))
    keywords = list(map(lambda x: maximal_marginal_relevance(
        x[0].reshape(1, -1),x[1],x[2],top_n=10,diversity=0.8),
        zip(model.encode(model_answers),
        model_candidate_emb,model_candidates)))
    t.append(time.perf_counter() - t1)

    print("model keywords emb ...")
    t1 = time.perf_counter()
    keywords_emb = list(map(lambda x: get_words_emb(x),keywords))
    t.append(time.perf_counter() - t1)

    s = list(map(lambda model_emb: 
        list(map(lambda doc_emb: cos_sim(model_emb,doc_emb).__array__().max(axis=1), docs_keywords_emb)),keywords_emb))
    
    # save the results
    np.save(f'data/results/essaySet_{essay}_keywords_scores',np.array(s))
    times[essay] = t
    clear_output()