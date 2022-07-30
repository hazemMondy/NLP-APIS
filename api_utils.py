"""api_utils.py"""

from typing import Optional, Dict, Tuple

def dict_to_list(
    dictionary: Dict[int or str, str], cased: Optional[bool]=False)\
    -> Tuple[list[str], list[str]]:
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

    example:
        >>> dict_to_list({1:"A",2:"B",3:"c"}, True)
        ([1,2,3],["A","B","c"])
    """
    keys , vals = list(map(str, list(dictionary.keys()))), list(dictionary.values())
    if cased:
        return keys, vals
    # convert every element to str
    return keys, list(map(lambda x: str(x).lower(), vals))
