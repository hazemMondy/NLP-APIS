"""api"""
from typing import Dict, List
import json
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import models.transformer_model as transformer_model
import models.keywords_grading as keywords_grading
import models.manual_keywords_grading as manual_keywords_grading
import models.siamese_ner_grading as siamese_ner_grading
import models.plagiarism_model as plagiarism_model
from utils.api_utils import dict_to_list
from configs.configs import configs as cfg

DEBUGGING = cfg["debugging"]
if not DEBUGGING:
    import warnings
    warnings.filterwarnings('ignore')

BERT = transformer_model.BERTModel()
# grading models
SNGM = siamese_ner_grading.SIAMESENERGradingModel(BERT)
KGM = keywords_grading.KeywordsGradingModel(BERT)
MKGM = manual_keywords_grading.ManualKeywordsGradingModel(BERT)
MODELS = { "siamese_ner_grading": SNGM, "automatic_keywords_grading": KGM, "manual_keywords_grading": MKGM }
# plagiarism model
PM = plagiarism_model.PlagiarismModel(BERT)

app = FastAPI(
    title="automatic Essay Grading and Pligarism check Models API",
    description="An API that use NLP models to return the plagiarism , grading result",
    version="2.0.1",
)

class StudentsDict(BaseModel):
    """
    StudentsDict input model validation

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the input

    Attributes:
        essays_dict (dict): essays_dict
            contains dicts of the student's answers and their ids
        cased (bool): cased strings grading or not

    example:
        >>> "essays_dict": {
        >>>  "1235dx":
        >>>        "answer_1"
        >>>    "24463dxcf":
        >>>         "answer_2"
        >>> }
        >>> "cased": False
    """
    essays_dict: Dict[str, str]
    cased: bool = False
class ESSAYSDICT(BaseModel):
    """
    ESSAYSDICT input model validation

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the input

    Attributes:
        essays_dict (dict): essays_dict
            contains dicts of the student's answers and their ids
        grading_model (str): grading_model
            grading model to use choose from ["siamese_ner_grading",
                "keywords_grading" , "manual_keywords_grading"]
        cased (bool): cased strings grading or not

    example:
        >>> "essays_dict": {
        >>>  "1235dx":
        >>>        "answer_1"
        >>>    "24463dxcf":
        >>>         "answer_2"
        >>> }
        >>> "grading_model": "keywords_grading"
        >>> "cased": False
    """
    essays_dict: Dict[str, str]
    grading_model: str = "siamese_ner_grading"
    cased: bool = False

    # validation of the grading_model to have only the 3 options
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.grading_model not in ["siamese_ner_grading", "automatic_keywords_grading",
             "manual_keywords_grading"]:
            raise ValueError("grading_model must be one of the following: \
                siamese_ner_grading, automatic_keywords_grading, manual_keywords_grading")
class PlagiarismResponse(BaseModel):
    """
    PlagiarismResponse output model validation
        it's a json string contains a List of Dictionaries of str [ids] : float [scores]

    Args:
        BaseModel : inherit from pydantic BaseModel to validate the output

    Attributes:
        List[ Dict[str, Dict[str,float]]]
            List of Dictionaries of str [ids] : float [scores]

        scores (List[Dict[str, float]]): scores
            contains a list of dicts of the student's answers and their ids

    example:
        >>> "plagiarism_results": [
        >>>     {"1235dx": 0.9, "24463dxcf": 0.8},
        >>>     {"1235dx": 0.8, "24463dxcf": 0.9}
        >>> ]
    """
    plagiarism_results : List[ Dict[str, Dict[str,float]]]
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
        >>> "essays_dict": {
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
def predict_grad(answers: ESSAYSDICT)\
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
        >>> predict_grad("essays_dict":
        >>> {"7821": "ahmed is eating a pizza",
        >>> "156": "i go to school by bus",
        >>> "3": "i am eating a pizza",
        >>> "4": "ahmed is eating a pizza"}
        >>>  "grading_model": "automatic_keywords_grading")

        "grades": {
            "156": 0.028096482157707214,
            "3": 0.45442324082056684,
            "4": 0.9999998410542806
                }
    """
    model = MODELS[answers.grading_model]
    ids, answers = dict_to_list(answers.essays_dict, answers.cased)
    res = model.predict(ids=ids, answers=answers)
    return {"grades":res}

@app.post("/plagiarism/predict", response_model=PlagiarismResponse)
def predict_plagiarism(essays_dict: StudentsDict)\
    -> List[ Dict[str, Dict[str,float]]]:
    """
    Predict the pligarism result

    Args
    ----------
    essays_dict : dict[str,str]
        dict of student ids and their answers

    Returns
    -------
    dict[str,dict[str,float]]
        dict of cheating studetns ids
        and their probabilities of plagiarism with other students

    example
    -------
    >>> predict_plagiarism('{1:\"Hello World\", 2:\"Hello World\"}, 3:\"hi there\"},cased: False')
    >>> [{1: {2: 0.98}, 2: {1: 0.98}}]

    """
    ids, answers = dict_to_list(essays_dict.essays_dict, essays_dict.cased)
    res = PM.predict(answers, ids)
    if DEBUGGING:
        print(res)
    return {"plagiarism_results":res}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
