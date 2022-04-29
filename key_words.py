"""key_words.py"""

from typing import Optional, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sentence_transformers.util import cos_sim as cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

def maximal_marginal_relevance(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: Optional[int] = 5,
        diversity: Optional[float] = 0.8)\
            -> List[Tuple[str, float]]:
    """
    Maximal Marginal Relevance algorithm for keyword extraction

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
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

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

def candidates_tokens(
    doc:str,
    n_gram_range : Optional[
        Tuple[int,int]]= (1,1))\
    -> List[str]:
    """
    extract candidates from a document

    Args:
        doc (str): document
        n_gram_range (Optional[Tuple[int,int]]): n_gram range

    Returns:
        List[str]: list of candidates words/phrases
    """

    stop_words = "english"
    # Extract candidate words/phrases
    count = CountVectorizer(
        ngram_range=n_gram_range,
        stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names()
    return candidates

def get_n_grams(
    keywords:List[str])\
    -> List[Tuple[int,int]]:
    """
    get n_gram range for each keyword

    Args:
        keywords (List[str]): list of keywords

    Returns:
        List[Tuple[int,int]]: list of n_gram range
    """
    # get ngrams for each keyword len
    # range ( len , len +1)
    n_gram_ranges = list(
        map(
            lambda word : (
                len(word.split()),
                len(word.split())+1
                ),
                keywords))

    return n_gram_ranges

def get_candidates(
    n_gram_ranges: List[
        Tuple[int,int]],
    paragraph:str)\
        -> List[List[str]]:
    """
    get candidates for each keyword grams

    Args:
        n_gram_ranges (List[Tuple[int,int]]): list of n_gram range
        paragraph (str): paragraph

    Returns:
        List[List[str]]: list of candidates
    """
    # candidates = key_words.candidates_tokens(paragraph, n_gram_range=n_gram_range[0])
    # * paralel processing to get candidates
    candidates = list(map( lambda gram :
        candidates_tokens(paragraph, n_gram_range=gram)
        , n_gram_ranges ))
    return candidates


def match_keywords(
    keywords_emb:List[np.ndarray],
    candidates_emb : List[np.ndarray],
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
    """

    # * paralel processing to get candidates
    # * retrive embs for testing

    # ! for testing
    # candidates_emb_shapes = list(map(lambda emb: emb.shape, candidates_emb))
    # keys_emb_shapes = list(map(lambda emb: emb.reshape(1, -1).shape, keywords_emb))

    combination = list(zip(keywords_emb,candidates_emb))

    similarities = list(map(lambda comb:
                    cosine_similarity(comb[0],
                    comb[1]),
                    combination ))

    def fn_ (x_x: np.array):
        """
        return the no. of matched keywords
        """
        if not np.sum(x_x>= thershold):
            return 0
        if np.sum(x_x >= thershold) > 1.0:
            return 1.0
        return np.sum(x_x >= thershold)

    res = np.sum(
        np.array(
            list(
                map(
                    fn_, similarities
                    )
                )
            )
            )

    return res/float(len(keywords_emb))
