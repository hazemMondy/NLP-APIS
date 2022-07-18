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
    keys , vals = list(dictionary.keys()), list(dictionary.values())
    # convert every element to str
    return list(map(lambda x: str(x).lower(), keys)), list(map(lambda x: str(x).lower(), vals))