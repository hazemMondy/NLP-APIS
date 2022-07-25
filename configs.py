"""configs.py"""

import numpy as np

THRESHOLD_SOFT = 0.7
THRESHOLD_MEDIUM = 0.9
ENCLOSURE_SOFT = "\"\"@"
ENCLOSURE_MEDIUM = "\"\"@@"
ENCLOSURE_HARD = "\"\""
DEBUGGING = False
DEFAULT_WEIGHT = np.nan
EXCEPTION_ENTITES = set(["DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL", "CARDINAL"])
NER_PATH = 'en_core_web_sm'
SIAMESE_NER_MODEL_PATH = 'models/model_rg.pkl.pickle'
KWRDS_MODEL_PATH = 'models/kwrds_model.pickle'

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
    'exception_entities': EXCEPTION_ENTITES,
    'ner_path': NER_PATH,
    'siamese_ner_model_path': SIAMESE_NER_MODEL_PATH,
    'kwrds_model_path': KWRDS_MODEL_PATH
    }
