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
    """
    students_dict: Dict[int, str]

class JsonDictionary(BaseModel):
    """
    StudentsDict output model validation
        it's a json string contains a List of Dictionaries of int [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output
    """
    plagiarism_results : str

class NormalDictionary(BaseModel):
    """
    StudentsDict output model validation
        it's a json string contains a List of Dictionaries of int [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output
    """
    plagiarism_results : List[
        Dict[int,
            Dict[int,float]
        ]]


class GradingResponse(BaseModel):
    """
    StudentsDict output model validation
        it's a json string contains a List of Dictionaries of int [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output
    """
    grads : Dict[int,float]

@app.get(path="/")
def read_root(query: str):
    """
    read index (root)
    for testing purpose

    Args:
        query (str): query string

    Returns:
        str: \"message\": query
    """

    return json.dumps({"message": query})


@app.post("/grading/predict", response_model=GradingResponse)
def predict_grad(answers: StudentsDict)\
     -> GradingResponse:
    """
    Predict the grading result

    Args
    ----------
    answers : dict[int,str]
        dict of student ids and their answers
        first element is the model answer
        second etc... elements are the students answers

    Returns
    -------
    Dict[int,float]
        dict of studetns ids and their grades

    example
    -------
        >>> predict_grad("students_dict":
            {"7821": "ahmed is eating a pizza",
            "156": "i go to school by bus",
            "3": "i am eating a pizza",
            "4": "ahmed is eating a pizza"})

        >>> "grads": {
            "156": 0.028096482157707214,
            "3": 0.45442324082056684,
            "4": 0.9999998410542806
                }
    """
    ids, answers = dict_to_list(answers.students_dict)
    print(ids)
    res = GM.predict(answers, ids)
    return {"grads":res}


@app.post("/plagiarism/predict", response_model=NormalDictionary)
def read_predict(students_dict: StudentsDict)\
    -> List[
        Dict[int,
            Dict[
                int,float
                ]]]:
    """
    Predict the pligarism result

    Args
    ----------
    students_dict : dict[int,str]
        dict of student ids and their answers

    Returns
    -------
    dict[int,dict[int,float]]
        dict of cheating studetns ids
        and their probabilities of plagiarism with other students

    example
    -------
    >>> read_predict('{1:\"Hello World\", 2:\"Hello World\"}, 3:\"hi there\"}')
    >>> [{1: {2: 0.98}, 2: {1: 0.98}}]

    """
    print(students_dict.students_dict)
    ids, answers = dict_to_list(students_dict.students_dict)
    res = PM.predict(answers, ids)
    return {"plagiarism_results":res}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=80)
