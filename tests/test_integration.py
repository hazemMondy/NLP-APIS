"""API testing"""

import pickle
import requests as req

PLAGIARISM_URL = "http://127.0.0.1:8000/plagiarism/predict/"
GRADING_URL = "http://127.0.0.1:8000/grading/predict/"

def load_from_pickle(filename):
    with open(filename, 'rb') as file_:
        return pickle.load(file_)

# load results from pickle file
grading_inp = load_from_pickle("tests/data/grading_inp.pickle")
plagiarism_inp = load_from_pickle("tests/data/plagiarism_inp.pickle")
grading_inp_err = load_from_pickle("tests/data/grading_inp_err.pickle")
plagiarism_inp_err = load_from_pickle("tests/data/plagiarism_inp_err.pickle")

class TestMainPredictPlagiarism(object):
    errors = ["value_error" ,"type_error", "key_error","TypeError", "ValueError", "KeyError"]
    def test_predict_plagiarism_normal(self):
        for inp in plagiarism_inp:
            actual = req.post(PLAGIARISM_URL, json=inp)
            message = f"actual type is: '{actual}', expected: 'list'"
            assert isinstance(actual.json()['plagiarism_results'], list), message

    def test_predict_plagiarism_errors(self):
        for inp in plagiarism_inp_err:
            actual = req.post(PLAGIARISM_URL, json=inp)
            actual_err = actual.json()["detail"][0]["type"]
            message = f"actual error is: '{actual_err}', expected: '{self.errors}'"
            assert actual_err in self.errors or self.errors[1] in actual_err, message

class TestAPIPredictGrading:
    errors = ["value_error" ,"type_error", "key_error","TypeError", "ValueError", "KeyError"]
    def test_predict_grading_normal(self):
        for inp in grading_inp:
            actual = req.post(GRADING_URL, json=inp)
            message = f"actual type is: '{actual.json()}', expected: 'list'"
            assert isinstance(actual.json()['grades'], dict), message

    def test_predict_grading_errors(self):
        for inp in grading_inp_err:
            actual = req.post(GRADING_URL, json=inp)
            actual_err = actual.json()["detail"][0]["type"]
            message = f"actual error is: '{actual_err}', expected: '{self.errors}'"
            assert actual_err in self.errors or self.errors[1] in actual_err, message
