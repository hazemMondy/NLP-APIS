"""test data generation"""

import pickle
import random
import string
from random_word import RandomWords

GRADING_MODEL = ["siamese_ner_grading", "automatic_keywords_grading","manual_keywords_grading"]
r = RandomWords()
ntests = 3
INPS = []

inp={   "78vv74": "\"\"ahmed\"\" is eating a pizza",
        "15xzx6": "i go to school by bus",
        "3": "i am eating a pizza",
        "4aiyfgd": "ahmed is eating a pizza",
        "15x": "i go to school by bus",
        "784":"i go to school by bus",
        "456y":"i am google",
        "7nqw1":"pv=nrt^2 /89641"}
inp2 = {"7er":"1234","7erqwr":"[1,2,3,4]",}
inp3 = {
        "78vv74": "\"\"@@as@@\"\"1.0 \"\"how\"\" " ,
        "9": "name all is good asdf dafdr increase",
        "10": "all is good how dafdr",
        "45": "all is asfy name",
        "451": "all is askdfhg how name",
        "5": "all is iopipo name",
        "4": "all is 782sd fjhv name"}
inp4 = {"78vv74": "",
        "15xzx6": "i go to school by bus",}

# generate random english text
def random_string(length:int=10):
    # return str(''.join(random.choices(string.ascii_lowercase, k=length)))
    return str(r.get_random_word())

def random_sentence(words:int=10,
    # word_length:int=8
    ):
    # return str(' '.join([random_string(word_length) for i in range(words)]))
    return str(' '.join([random_string() for i in range(words)]))

def random_ids(n:int=10):
    return [str(''.join(random.choices(string.hexdigits + string.ascii_lowercase + string.digits, k=i%5 +1))) for i in range(n)]

def generate_inp(answers:int=4, words=10,
    # word_length=8
    ):
    ids = random_ids(answers)
    answers = [random_sentence( words%answers + words//3) for i in range(answers)]
    return zip(ids,answers)

def get_plagiarism_inp(inps= None
    # cased:bool=False
    ):
    if inps is None:
        inps = INPS
    for i in inps:
        yield {
            "students_dict":i,
            "cased":random.choice([True,False])
            }

def get_grading_inp(inps= None
    # grading_model:str,cased:bool=False
    ):
    if inps is None:
        inps = INPS
    for i in inps:
        yield {
            "essays_dict":i,
            "grading_model":random.choice(GRADING_MODEL),
            "cased":random.choice([True,False])
            }

err_inps_plagiarism = [{
        "78vv74": None,
        "15xzx6": "i go to school by bus",
        },
        {
        "78vv74": [],
        "15xzx6": "i go to school by bus",
        },
        {
        ("78vv74",): "dfdfadf",
        "15xzx6": "i go to school by bus",
        },
]

err_inps_grading = [{
        "78vv74": "asdasf",
        "15xzx6": "i go to school by bus",},
        ]*4

def get_grading_inp_err(inps=err_inps_grading):
    models = ["asd"]+GRADING_MODEL
    for i,v in enumerate(inps):
        yield {
            "essays_dict":v,
            "grading_model":models[i%len(models)],
            "cased":random.choice([False,[],21])
            }

def get_plagiarism_inp_err(inps=err_inps_plagiarism):
    for i,v in enumerate(inps):
        yield {
            "students_dict":v,
            "cased":random.choice([False,True])
            }

# save results to pickle file
def save_to_pickle(obj,filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
def load_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

INPS = [dict(generate_inp(random.randint(3, 12),random.randint(4, 12))) for i in range(ntests)]+ [inp]+ [inp2]+ [inp3]
grading_inp = list(get_grading_inp())
plagiarism_inp = list(get_plagiarism_inp())
grading_inp_err = list(get_grading_inp_err())
plagiarism_inp_err = list(get_plagiarism_inp_err())

save_to_pickle(grading_inp_err,"grading_inp_err.pickle")
save_to_pickle(plagiarism_inp_err,"plagiarism_inp_err.pickle")
save_to_pickle(INPS,"inps.pickle")
save_to_pickle(grading_inp,"grading_inp.pickle")
save_to_pickle(plagiarism_inp,"plagiarism_inp.pickle")
