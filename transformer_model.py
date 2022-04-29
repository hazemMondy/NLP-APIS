"""transformer_model.py"""

from typing import Optional
from sentence_transformers import SentenceTransformer

MODELPATH = 'sentence-transformers/xlm-r-distilroberta-base-paraphrase-v1'

class BERTModel(object):
    """BERTModel"""
    model:object = None
    def __new__(
        cls: object,
        modelpath: Optional[str]=MODELPATH):
        if not hasattr(cls, 'instance'):
            cls.instance = super(BERTModel, cls).__new__(cls)
            cls.model = SentenceTransformer(modelpath)
        return cls.instance
