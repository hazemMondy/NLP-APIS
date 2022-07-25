"""utils.utils.py"""

from typing import List, Any
import pickle
import enum

class KeywordType(enum.Enum):
    """
    KeywordType
    """
    SOFT = enum.auto()
    MEDIUM = enum.auto()
    HARD = enum.auto()
    def __str__(self):
        return self.name.lower()
    def __repr__(self):
        return self.name.lower()

def flatten(ls_ls:List[List[Any]])->List[Any]:
    """
    Flatten a list of lists.

    Args:
        ls_ls (List[List[Any]]): list of lists

    Returns:
        List[Any]: flattened list

    example:
        >>> flatten([[1,2],[3,4]])
        [1,2,3,4]
    """
    out = []
    for lis in ls_ls:
        if isinstance(lis, list):
            out.extend(lis)
        else:
            out.append(lis)
    return out

def save_obj(obj:object, name:str, ext = '.pickle', path=None):
    """
    Save an object to a file.

    Args:
        obj (object): object to save
        name (str): name of the file
        ext (Optional[str]): extension of the file

    Returns:
        None

    example:
        >>> save_obj(doc_list,'doc_list')
    """
    if path is None:
        path = name + ext
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(name:str, ext = '.pickle', path=None)->object:
    """
    Load an object from a file.

    Args:
        name (str): name of the file
        ext (Optional[str]): extension of the file

    Returns:
        object: loaded object

    example:
        >>> doc_list = load_obj('doc_list')
    """
    if path is None:
        path = name + ext
    with open(path, 'rb') as handle:
        return pickle.load(handle)
