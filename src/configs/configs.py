"""configs.py"""

import numpy as np

THRESHOLD_SOFT = 0.7
THRESHOLD_MEDIUM = 0.9
ENCLOSURE_SOFT = "\"\"@"
ENCLOSURE_MEDIUM = "\"\"@@"
ENCLOSURE_HARD = "\"\""
DEBUGGING = False
DEFAULT_WEIGHT = np.nan
DEFAULT_EMPTY_WEIGHT = np.NINF
EXCEPTION_ENTITES = set(["DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL", "CARDINAL"])
NER_PATH = 'en_core_web_sm'
# D:\projects\NLP-APIS\models\kwrds_model.pickle
SIAMESE_NER_MODEL_PATH = 'D:/projects/NLP-APIS/models/model_rg.pickle'
KWRDS_MODEL_PATH = 'D:/projects/NLP-APIS/models/kwrds_model.pickle'
ENCODER_PATH = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

configs = {
    'debugging': DEBUGGING,
    'port': 8080,
    'host': '',
    'enclosure_soft': ENCLOSURE_SOFT,
    'enclosure_medium': ENCLOSURE_MEDIUM,
    'enclosure_hard': ENCLOSURE_HARD,
    'threshold_soft': THRESHOLD_SOFT,
    'threshold_medium': THRESHOLD_MEDIUM,
    'default_weight': DEFAULT_WEIGHT,
    'default_empty_weight': DEFAULT_EMPTY_WEIGHT,
    'exception_entities': EXCEPTION_ENTITES,
    'ner_path': NER_PATH,
    'siamese_ner_model_path': SIAMESE_NER_MODEL_PATH,
    'encoder_path': ENCODER_PATH
    }
# keywords_grading.py
DEFAULT_PADDING_LENGTH = 10 # coefficient of the neural network's input length
DEFAULT_PADDING_VALUE = ""
TOP_N_KEYWORDS = 10 # number of keywords to extract from an essay
DIVERSITY = 0.7
N_GRAMS = [(2,3)]
BATCH_SIZE = 5

configs_keywords = {
    'debugging': DEBUGGING,
    'kwrds_model_path': KWRDS_MODEL_PATH,
    'padding_length': DEFAULT_PADDING_LENGTH,
    'padding_value': DEFAULT_PADDING_VALUE,
    'top_n_keywords': TOP_N_KEYWORDS,
    'diversity': DIVERSITY,
    'n_grams': N_GRAMS,
    'batch_size': BATCH_SIZE
    }
