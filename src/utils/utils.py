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

def flatten(ls_ls:List[List[Any]], degree=2)->List[Any]:
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
    if not isinstance(ls_ls, list) and not None:
        raise TypeError("input must be a list")
    if degree is None or degree < 1:
        degree = 2

    def _flatten(ls:List[Any])->List[Any]:
        out = []
        for lis in ls:
            if isinstance(lis, list):
                out.extend(lis)
            else:
                out.append(lis)
        return out

    for _ in range(degree-1):
        ls_ls = _flatten(ls_ls)

    return ls_ls

def save_obj(obj:object, name:str, ext = '.pickle', path=None):
    """
    Save an object to a file.

    Args:
        obj (object): object to save
        name (str): name of the file
        ext (Optional[str]): extension of the file
        path (Optional[str]): full path of the file with name

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
        path (Optional[str]): full path of the file with name

    Returns:
        object: loaded object

    example:
        >>> doc_list = load_obj('doc_list')
    """
    if path is None:
        path = name + ext
    with open(path, 'rb') as handle:
        return pickle.load(handle)
