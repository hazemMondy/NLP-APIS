from typing import Optional, Dict, Tuple
def dict_to_list(
    dictionary:Dict[int,str])\
    ->Tuple[list,list]:
    """
    Convert dict to list

    Parameters
    ----------
    d : dict[int,str]
        dict to convert

    Returns
    -------
    tuple[list,list]
        list of keys and values
        keys are ids
        values are text (answers)
    """
    return list(dictionary.keys()), list(dictionary.values())

import sys

print(sys.path)