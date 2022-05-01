"""api"""
from typing import Optional, Dict, List
import json
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from api_utils import dict_to_list
import grading_model, transformer_model, plagiarism_model

BERT = transformer_model.BERTModel()

GM = grading_model.GradingModel(BERT)
PM = plagiarism_model.PlagiarismModel(BERT)

app = FastAPI(
    title="Essay Grading and Pligarism Model API",
    description="An API that use NLP model to return the plagiarism , grading result",
    version="1.2",
)

class StudentsDict(BaseModel):
    """
    StudentsDict input model validation

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the input
    
    Attributes:
        students_dict (dict): students_dict
            contains dicts of the student's answers and their ids
    
    example:
        >>> "students_dict": {
        >>>  "1235dx":
        >>>        "answer_1"
        >>>    "24463dxcf": 
        >>>         "answer_2"
        >>> }


    """
    students_dict: Dict[str, str]

class PlagiarismResponse(BaseModel):
    """
    PlagiarismResponse output model validation
        it's a json string contains a List of Dictionaries of str [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output
    
    Attributes:
        scores (List[Dict[str, float]]): scores
            contains a list of dicts of the student's answers and their ids

    example:
        >>> {
        >>>     "ids": ["1", "2", "3"],
        >>>     "scores": [0.5, 0.5, 0.5]
        >>> }
    """
    plagiarism_results : List[
        Dict[str,
            Dict[str,float]
        ]]


class GradingResponse(BaseModel):
    """
    StudentsDict output model validation
        it's a json string contains a List of Dictionaries of str [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output
    
    Attributes:
        grades ([Dict[str, float]]): scores
            contains a list of dicts of the student's grades and their ids

    example:
        >>> "students_dict": {
        >>>  "1235dx":
        >>>        "answer_1"
        >>>    "24463dxcf": 
        >>>         "answer_2"
        >>> }

        >>> "grades":
        >>>     {
        >>>         "7818ert": 1,
        >>>         "4581rdc": 0.5,
        >>>     }

    """
    grades : Dict[str,float]

@app.get(path="/")
def read_root():
    """
    read index (root)
    for testing purpose

    Args:
        query (str): query string

    Returns:
        str: \"message\": hi
    """

    return json.dumps({"message": "Hi"})


@app.post("/grading/predict", response_model=GradingResponse)
def predict_grad(answers: StudentsDict)\
     -> GradingResponse:
    """
    Predict the grading result

    Args
    ----------
    answers : dict[str,str]
        dict of student ids and their answers
        first element is the model answer
        second etc... elements are the students answers

    Returns
    -------
    Dict[str,float]
        dict of studetns ids and their grades

    example
    -------
        >>> predict_grad("students_dict":
            {"7821": "ahmed is eating a pizza",
            "156": "i go to school by bus",
            "3": "i am eating a pizza",
            "4": "ahmed is eating a pizza"})

        >>> "grades": {
            "156": 0.028096482157707214,
            "3": 0.45442324082056684,
            "4": 0.9999998410542806
                }
    """
    ids, answers = dict_to_list(answers.students_dict)
    res = GM.predict(answers, ids)
    return {"grades":res}


@app.post("/plagiarism/predict", response_model=PlagiarismResponse)
def read_predict(students_dict: StudentsDict)\
    -> List[
        Dict[str,
            Dict[
                str,float
                ]]]:
    """
    Predict the pligarism result

    Args
    ----------
    students_dict : dict[str,str]
        dict of student ids and their answers

    Returns
    -------
    dict[str,dict[str,float]]
        dict of cheating studetns ids
        and their probabilities of plagiarism with other students

    example
    -------
    >>> read_predict('{1:\"Hello World\", 2:\"Hello World\"}, 3:\"hi there\"}')
    >>> [{1: {2: 0.98}, 2: {1: 0.98}}]

    """
    ids, answers = dict_to_list(students_dict.students_dict)
    res = PM.predict(answers, ids)
    print(res)
    return {"plagiarism_results":res}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
