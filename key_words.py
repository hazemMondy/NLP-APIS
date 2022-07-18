"""key_words.py"""

from typing import Optional, List, Tuple
import regex as re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers.util import cos_sim
# from sklearn.metrics.pairwise import cosine_similarity

def maximal_marginal_relevance(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: Optional[int] = 5,
        diversity: Optional[float] = 0.8)\
            -> List[Tuple[str, float]]:
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
    # word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    # word_similarity = cosine_similarity(word_embeddings)

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
    candidates = [doc]
    try:
        count = CountVectorizer(
            ngram_range=n_gram_range,
            stop_words=stop_words).fit([doc])
        candidates = count.get_feature_names()
    except Exception as e:
    # except Exception("empty vocabulary") as e:
        stop_words
    
    # * FASTER
    # unique candidates
    return [*set(candidates)]

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
    # * removed redundents
    n_gram_ranges = [*set(list(
        map(lambda word : (
            len(word.split()),
            len(word.split())+1),
            keywords)))]
    return sorted(n_gram_ranges)

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
    candidates = list(map(lambda gram :
        candidates_tokens(str(paragraph), n_gram_range=gram)
        , n_gram_ranges))
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

    example:
        >>> match_keywords(keywords_emb, candidates_emb, thershold=0.5)
        >>> 0.8
    """

    # * paralel processing to get candidates
    # * retrive embs for testing

    # ! for testing
    # candidates_emb_shapes = list(map(lambda emb: emb.shape, candidates_emb))
    # keys_emb_shapes = list(map(lambda emb: emb.reshape(1, -1).shape, keywords_emb))
    # check if candidates_emb shape is 1d
    def shape_check(emb):
        if emb.size == 1:
            return emb.reshape(-1, 1)
        else:
            return emb
    
    candidates_emb = list(map(shape_check, candidates_emb))
    keywords_emb = list(map(shape_check, keywords_emb))

    combination = list(zip(keywords_emb,candidates_emb))

    similarities = list(map(lambda comb:
                    np.array(cos_sim(comb[0],
                    comb[1])).clip(-1, 1).round(6),
                    combination))

    def fn_ (x: np.array):
        """
        return the no. of matched keywords
        """
        if not np.sum(x>= thershold):
            return 0
        if np.sum(x >= thershold) > 1.0:
            return 1.0
        return np.sum(x >= thershold)

    # res = np.sum(np.array(list(map(fn_, similarities))))
    # 5.51 ms ± 59.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) for 600 elements

    res = sum(map(fn_, similarities))
    # 5.45 ms ± 52.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) for 600 elements
    # return res/float(len(keywords_emb))
    return res/len(keywords_emb)

def reverse_string(string:str)->str:
    return string[::-1]

def get_str_between(doc:str,enclosure:str="\"\"")->List[str]:
    """
    get string between two enclosures

    Args:
        doc (str): document
        enclosure (str): enclosure

    Returns:
        List[str]: list of strings

    example:
        >>> doc = "\"hello\" \"world\""
        >>> get_str_between(doc)
        >>> ["hello", "world"]
    """
    # reverse a string
    return re.findall(r'{}(.*?){}'.format(enclosure,reverse_string(enclosure)), doc)


def match_keywords_in_doc(keywords:List[str],doc:str):
    """
    match keywords with candidates in a document

        Args:
            keywords (List[str]): list of keywords
            doc (str): document

        Returns:
            float: score

        example:
            >>> match_keywords_in_doc(["people", "theory"], "people of africa")
            >>> 0.5
    """
    return sum(map(lambda keyword:keyword in doc,keywords))

def hard_keywords_grading(keywords:List[str],docs:List[str]):
    """
    hard keywords grading

    Args:
        keywords (List[str]): list of keywords
        docs (List[str]): list of docs

    Returns:
        np array double: scores

    example:
        >>> hard_keywords_grading(["people", "theory"], ["people are awesome", "theory", "science"])
        >>> array([0.5, 0.5, 0.])
    """
    if not isinstance(keywords,list):
        keywords = [keywords]
    return np.array(list(map(lambda doc :match_keywords_in_doc(keywords, doc),docs))) / len(keywords)
