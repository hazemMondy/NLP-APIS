"""strings_utils.py"""

from typing import List
import regex as re
import numpy as np
from configs.configs import configs as cfg

DEFAULT_WEIGHT = cfg['default_weight']
DEFAULT_EMPTY_WEIGHT = cfg['default_empty_weight']
ENCLOSURE_HARD = cfg['enclosure_hard']
def reverse_string(doc:str)->str:
    """
    Reverse a string.

    Args:
        doc (str): string to reverse

    Returns:
        str: reversed string

    example:
        >>> reverse_string('abc')
        'cba'
    """
    if doc is None:
        return None

    if not isinstance(doc, str):
        raise TypeError("doc must be a string")

    return doc[::-1]

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
    if doc is None:
        return []

    if enclosure is None:
        enclosure = ENCLOSURE_HARD

    if not isinstance(doc, str):
        raise TypeError("doc must be a string")

    if not isinstance(enclosure, str):
        raise TypeError("enclosure must be a string")

    if len(enclosure) == 0:
        raise ValueError("enclosure must be a valid non-empty string")

    return re.findall(r'{}(.*?){}'.format(enclosure,reverse_string(enclosure)), doc)

#! *****************************************************************************
def match_keywords_in_doc(keywords:List[str],doc:str)->np.ndarray:
    """
    match keywords with candidates in a document

        Args:
            keywords (List[str]): list of keywords
            doc (str): document

        Returns:
            np.ndarray of int: array of matches

        example:
            >>> match_keywords_in_doc(["people", "theory"], "people of africa")
            array([1, 0])
    """
    default_out = np.zeros((1,))

    if keywords is None:
        return default_out

    if len(keywords) == 0:
        return default_out

    if not isinstance(keywords, list):
        keywords = [keywords]

    if not isinstance(keywords[0], str):
        raise TypeError("keywords must be a list of strings")

    default_out = np.zeros((len(keywords),))

    if doc is None:
        return default_out

    if not isinstance(doc, str):
        raise TypeError("doc must be a string")

    if len(doc) == 0:
        return default_out

    return np.array(list(map(lambda keyword:keyword in doc,keywords))).astype(int)
    # return sum(map(lambda keyword:keyword in doc,keywords))

def clean_doc(doc:str, keywords:List[str], weights:List[float], enclosure:str)->str or None:
    """
    clean document

    Args:
        doc (str): document
        keywords (List[str]): list of keywords
        weights (List[float]): list of weights
        enclosure (str): enclosure

    Returns:
        str: cleaned document

    example:
        >>> clean_doc("\"\"hello world\"\"1.0 papa", ["hello world"], [1.0], "\"\"")
        >>> "papa"
    """
    if doc is None or keywords is None:
        return doc
    if weights is None:
        raise ValueError("weights must be a list of floats")
    if enclosure is None:
        enclosure = ENCLOSURE_HARD

    if not isinstance(enclosure, str):
        raise TypeError("enclosure must be a string")

    if len(enclosure) == 0:
        raise ValueError("enclosure must be a valid non-empty string")

    if not isinstance(doc, str):
        doc = str(doc)

    if not isinstance(keywords,list):
        keywords = [keywords]

    for i,keyword in enumerate(keywords):
        # get the end index of the keyword in the doc
        # and the next word after the keyword
        key = enclosure + str(keyword) + reverse_string(enclosure)

        if weights[i] in [DEFAULT_EMPTY_WEIGHT, DEFAULT_WEIGHT]:
            doc = doc.replace(key,"")
            continue

        doc_ = doc
        cand = key + str(weights[i])
        doc_ = doc.replace(cand, "")
        if doc_ == doc:
            cand = key + str(int(weights[i]))
            doc_ = doc.replace(cand, "")
        doc = doc_
    # clean exta spaces
    doc = re.sub(r'\s+', ' ', doc)
    doc = re.sub(r'\s+', ' ', doc)
    return doc

def clean_punctuation(doc:str)->str:
    """
    clean document from punctuation

    Args:
        doc (str): document

    Returns:
        str: cleaned document

    example:
        >>> clean_doc_from_punctuation('You need \"\"vinegar0.6 @container. at 8º ')
        >>> 'You need vinegar06 container at 8º'
    """
    if doc is None:
        return None
    doc = re.sub(r'[^\w\s]','',str(doc))
    doc = re.sub(r'[^\w\s]','', doc)
    return doc

def parse_float(doc:str)->List[float] or []:
    """
    parse float from document

    Args:
        doc (str): document

    Returns:
        List[float]: list of floats

    example:
        >>> parse_float('You need \"\"vinegar\"\"0.6 @container. at 8º ')
        >>> [0.6]
    """
    if doc is None:
        return []
    if not isinstance(doc, str):
        raise TypeError("doc must be a string")

    return list(map(float, re.findall(r'\d+\.\d+',doc)))
    # return re.findall(r'\d+\.\d*', doc)

def clean_doc_keep_float(doc:str)->str or None:
    """
    clean document from punctuation and keep float numbers

    Args:
        doc (str): document

    Returns:
        str: cleaned document

    example:
        >>> clean_doc_from_punctuation('You need \"\"vinegar0.6 @container. at 8º ')
        >>> 'You need vinegar0.6 container at 8º'
    """
    if doc is None:
        return None
    if (isinstance(doc, str) or isinstance(doc, float) or isinstance(doc, int)):
        doc = str(doc)
    else:
        raise TypeError("doc must be a string")

    doc_punc = clean_punctuation(doc)
    floats = parse_float(doc)
    floats_punc = [clean_punctuation(number) for number in floats]
    floats_dict = dict(zip(floats, floats_punc))
    for float_n, float_punc in floats_dict.items():
        doc_punc = doc_punc.replace(str(float_punc), str(float_n))
    # clean exta spaces
    doc_punc = re.sub(r'\s+', ' ', doc_punc)
    return doc_punc
