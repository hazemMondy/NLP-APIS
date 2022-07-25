"""strings_utils.py"""

from typing import List
import regex as re
import numpy as np

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
    return np.array(list(map(lambda keyword:keyword in doc,keywords)))
    # return sum(map(lambda keyword:keyword in doc,keywords))

def clean_doc(doc:str, keywords:List[str], weights:List[float], enclosure:str):
    """
    clean document

    Args:
        doc (str): document

    Returns:
        str: cleaned document

    example:
        >>> clean_doc("hello world")
        >>> "hello world"
    """
    if not isinstance(keywords,list):
        keywords = [keywords]
    for i,keyword in enumerate(keywords):
        # get the end index of the keyword in the doc
        # and the next word after the keyword
        key = enclosure + keyword + reverse_string(enclosure)
        if weights[i] == np.nan:
            doc = doc.replace(key,"")
        else:
            doc = doc.replace(key + str(weights[i]), "")
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
    doc = re.sub(r'[^\w\s]','',doc)
    return doc

def parse_float(doc:str)->List[float] or []:
    """
    parse float from document

    Args:
        doc (str): document

    Returns:
        List[float]: list of floats

    example:
        >>> parse_float('You need \"\"vinegar0.6 @container. at 8º ')
        >>> [0.6]
    """
    if doc is None:
        return []
        # return None
    return re.findall(r'\d+\.\d+', doc)

def clean_doc_keep_float(doc:str)->str:
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
    doc_punc = clean_punctuation(doc)
    floats = parse_float(doc)
    floats_punc = [clean_punctuation(number) for number in floats]
    floats_dict = dict(zip(floats, floats_punc))
    for float_n, float_punc in floats_dict.items():
        doc_punc = doc_punc.replace(float_punc, float_n)
    return doc_punc
