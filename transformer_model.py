"""transformer_model.py"""

from typing import Optional
from sentence_transformers import SentenceTransformer
from configs import configs as cfg

# MODELPATH = 'xlm-r-distilroberta-base-paraphrase-v1'
MODELPATH = cfg["encoder_path"]
DEBUGGING = cfg["debugging"]
if not DEBUGGING:
    import warnings
    warnings.filterwarnings('ignore')
class BERTModel(object):
    """BERTModel"""
    def __new__(
        cls: object, modelpath: Optional[str]=MODELPATH):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BERTModel, cls).__new__(cls)
            cls.model = SentenceTransformer(modelpath)
        return cls.instance
