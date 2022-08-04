"""keywords_grading.py"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from sentence_transformers.util import cos_sim
import utils.key_words as key_words
from utils.utils import flatten,load_obj,save_obj
from configs.configs import configs_keywords as cfg

DEBUGGING = cfg['debugging']
# PATH_TO_LAST_LAYER: path to the last layer classifier, regressor ...
LAST_LAYER_PATH = cfg['kwrds_model_path']
PADDING_LENGTH = cfg['padding_length']
PADDING_VALUE = cfg['padding_value']
TOP_N_KEYWORDS = cfg['top_n_keywords']
DIVERSITY = cfg['diversity']
N_GRAMS = cfg['n_grams']
BATCH_SIZE = cfg['batch_size']
class KeywordsGradingModel(object):
    """KeywordsGradingModel"""
    # words_emb_dict = {}

    def __new__(cls: object,encoder_model: object):
        """
        Args:
            encoder_model: encoder_model
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(KeywordsGradingModel, cls).__new__(cls)
            try:
                cls.model = encoder_model.model
            except AttributeError as err:
                if DEBUGGING:
                    print(err)
                cls.model = encoder_model
            cls.last_layer = load_obj(name = "", path = LAST_LAYER_PATH)
            cls.words_emb_dict = {}
        return cls.instance

    def _get_word_emb(self:object,word:str):
        """
        get word embedding

        Args:
            word: str

        Returns:
            np.array of shape (768,) or (384,) depending on the encoder model

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> kgm._get_word_emb('hello')
            array([0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97])
        """
        if word in self.words_emb_dict:
            return self.words_emb_dict[word]
        else:
            self.words_emb_dict[word] = self.model.encode(word)
            return self.words_emb_dict[word]

    def _get_candidates(self:object,n_grams:List[Tuple[int,int]], doc:str)->List[str]:
        """
        Returns the candidates for the given n-grams.

        Args:
            n_grams: List of n-grams.
            doc: The document.

        Returns:
            List of candidates.

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> n_grams = [(1,2),(2,3)]
            >>> doc = 'You need to know how much vinegar was used in each container.'
            >>> kgm._get_candidates(n_grams,doc)
            ['know vinegar used', 'vinegar used container', 'need know vinegar']
        """
        return flatten(list(map(lambda gram :
            key_words.candidates_tokens(str(doc), n_gram_range=gram), n_grams)))

    def _emb_keywords(self:object,keywords:List[str])->np.ndarray:
        """
        Args:
            keywords: List of keywords.
            doc: The document.

        Returns:
            np.array of shape (docs_number,768) or (docs_number,384) depending on the encoder model

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> keywords = ['hello','world']
            >>> kgm.emb_keywords_(keywords)
            array([[0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97],
                   [0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97]])
        """
        return np.array(list(map(self._get_word_emb, keywords)))

    def keywords_exrtaction(self:object,
        docs:List[str],
        n_grams:List[Tuple[int,int]] = N_GRAMS,
        top_n:int = TOP_N_KEYWORDS,
        diversity:float = DIVERSITY,
        batch:int = BATCH_SIZE,
        *args, **kwargs):
        """
        keywords exrtaction pipeline

        Args:
            docs (List[str]): list of documents
            n_grams (Optional[List[Tuple[int, int]]]): list of n_grams = [(2,3)]
            top_n (Optional[int]): number of top words to extract = 10
            diversity (Optional[float]): diversity of top words to extract = 0.7
            batch (Optional[int]): batch size = 5

        Returns:
            List[List[str]]: list of top_n keyprhases

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> docs = ['You need to know how much vinegar was used in each container.']
            >>> n_grams = [(2,3)]
            >>> kgm.keywords_exrtaction(docs,n_grams)
            [['know vinegar used', 'vinegar used container', 'need know vinegar']]
        """
        if isinstance(docs, str):
            docs = [docs]
        n_docs = len(docs)
        if n_docs < batch:
            batch = n_docs
        # all model answers
        docs_keys_ls = []
        # do in batches
        for i in range(0,n_docs,batch):
            if DEBUGGING:
                print("Processing batch {}/{}".format(i//batch+1, (n_docs//batch)+1))
            docs_candidates = list(map(lambda doc: self._get_candidates(n_grams, doc),
                docs[i:i+batch]))
            # print("Extracting embeddings")
            docs_candidates_emb =  np.array(list(map(self._emb_keywords,docs_candidates)))
            # print("Matching keywords")
            docs_emb = self.model.encode(docs[i:i+batch])
            # print("Extracting keywords")
            docs_keywords = list(map(lambda doc: key_words.maximal_marginal_relevance(
                    doc[0].reshape(1, -1),doc[1],doc[2],top_n=top_n ,diversity=diversity),
                    zip(docs_emb,docs_candidates_emb,docs_candidates)))
            docs_keys_ls.extend(docs_keywords)

        if n_docs % batch != 0 and n_docs > batch:
            docs_candidates = list(map(lambda doc: self._get_candidates(n_grams, doc),
                docs[i+batch:]))
            docs_candidates_emb =  np.array(list(map(self._emb_keywords, docs_candidates)))
            docs_emb = self.model.encode(docs[i+batch:])
            docs_keywords = list(map(lambda doc: key_words.maximal_marginal_relevance(
                    doc[0].reshape(1, -1),doc[1],doc[2],top_n=top_n ,diversity=diversity),
                    zip(docs_emb,docs_candidates_emb,docs_candidates)))
            docs_keys_ls.extend(docs_keywords)
        return docs_keys_ls

    def keywords_grading(self:object,
        docs_keywords: List[List[str]],model_answer_keywords:List[List[str]])-> np.ndarray:
        """
        Args:
            docs_keywords (List[List[str]]): list of top_n keyprhases for each document
            model_answer_keywords (List[List[str]]): list of top_n keyprhases for each model answer

        Returns:
            np.array of shape (docs_number,top_n) containing the scores for
                each keyword for each document

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> docs_keywords = [['know vinegar used', 'vinegar used container',
                 'need know vinegar']]
            >>> model_answer_keywords = [['know vinegar used', 'vinegar used container',
                 'need know vinegar']]
            >>> kgm.keywords_grading(docs_keywords,model_answer_keywords)
            array([[[1.0000002 , 1.0000002 , 0.99999964]]], dtype=float32)
        """
        if DEBUGGING:
            print(model_answer_keywords)

        docs_keywords_emb = list(map(lambda kwrds: np.array(list(map(
                self._get_word_emb ,kwrds))),docs_keywords))
        keywords_emb = list(map(lambda kwrds: np.array(list(map(
                self._get_word_emb ,kwrds))),model_answer_keywords))

        return np.array(list(map(lambda model_emb:
                np.array(list(map(lambda doc_emb:
                cos_sim(model_emb,doc_emb).__array__().max(axis=1),
                docs_keywords_emb))),keywords_emb))).round(3).clip(0,1)

    def keywords_grading_predict(self:object,grades:np.ndarray):
        """
        Args:
            grades (np.ndarray): array of shape (docs_number,top_n)
            containing the scores for each keyword for each document

        Returns:
            np.array of shape (n_model_answers,docs_number) containing
                the scores for each keyword for each document

        example:
            >>> kgm = KeywordsGradingModel(BERTModel)
            >>> grades = np.array([[[0.761, 0.242, 0.263, 0.832, 0.437,
                0.259, 0.269, 0.314, 0.218,0.237],
                [0.509, 0.392, 0.472, 0.451, 0.924, 0.366, 0.24 , 0.314, 0.236,
                0.237]]], dtype=float32)
            >>> kgm.keywords_grading_predict(grades)
            array([[0.467, 0.401]])
        """
        if DEBUGGING:
            print(grades.shape, grades.T.shape)
        out = np.array(list(map(self.last_layer.predict ,grades))).round(3).clip(0,1)
        # out = np.array(list(map(lambda grade: self.last_layer.predict(grade.T),grades)))
        # .round(3).clip(0,1)
        #* as for one model answer only
        return out.ravel()

    def fit(self:object, x_train:np.ndarray, y_train:np.ndarray, *args, **kwargs):
        """
        finetune the last layer model on the data

        Args:
            X (np.ndarray): input data
            y_train (np.ndarray): target data
            args: additional arguments for the fit function
            kwargs: additional keyword arguments for the fit function

        example:
            >>> model.fit(x_train,y_train,max_iter=100,alpha=0.0001)
        """
        self.last_layer.fit(x_train, y_train, *args, **kwargs)
        print("Model finetuned", "you may save the model now")

    def save(self:object,
        path:str = None):
        """
        save the model to a file

        Args:
            path (str): path to save the model to
            if None then the model will be saved to the default path

        example:
            >>> model.save()
            Model saved Successfully to models/kwrds_model.pickle
        """
        if path is None:
            path = LAST_LAYER_PATH
        save_obj(obj = self.last_layer, name = "", path=path)
        print("Model saved Successfully to", path)

    def pad_keywords(self:object,
        keywords:List[str], padding='post')->List[str]:
        """
        pad the keywords to the same length

        Args:
            keywords (List[str]): list of keywords

        Returns:
            List[str]: list of keywords with the same length

        example:
            >>> model.pad_keywords([['know', 'vinegar', 'used', 'container', 'need']])
            [['know', 'vinegar', 'used', 'container', 'need', 'defaul_padding', 'defaul_padding',
                 'defaul_padding', 'defaul_padding', 'defaul_padding']]
        """
        for ind, keyword in enumerate(keywords):
            if padding == 'post':
                keywords[ind] = keyword + [PADDING_VALUE] * (PADDING_LENGTH - len(keyword))
            elif padding == 'pre':
                keywords[ind] = [PADDING_VALUE] * (PADDING_LENGTH - len(keyword)) + keyword
            else:
                raise ValueError("padding must be 'post' or 'pre'")
        return keywords

    def predict(
        self: object,
        answers: List[str],
        ids: List[str],
        top_n: Optional[int] = TOP_N_KEYWORDS,
        diversity: Optional[float] = DIVERSITY,
        n_gram_range: Optional[List[Tuple[int,int]]] = N_GRAMS,
        )-> Dict[str, float]:
        """
        the main function to process the answers and return the scores
        Args:
            answers (List[str]): list of answers
            ids (List[str]): list of ids of the corresponding answers
            top_n (Optional[int]): number of top keywords to extract = 10
            diversity (Optional[float]): diversity parameter for the keywords extraction = 0.7
            n_gram_range (Optional[List[Tuple[int,int]]]): list of n_grams to extract = [(2,3)]
                from the answers

        Returns:
            Dict[str, float]: dictionary of ids with their scores

        example:
            >>> kgm.predict(answers,ids,top_n=10,diversity=0.7,n_gram_range=(2,3))
            {'id1': 0.8, 'id2': 0.7}
        """
        if answers is None or ids is None:
            raise ValueError("answers and ids must be provided")
        if isinstance(answers, str):
            answers = [answers]
        if len(answers) != len(ids):
            raise ValueError("every answers should be paired with an id")
        if len(answers) <= 1:
            raise ValueError("there should be at least 1 answer and the model-answer")
        if top_n is None:
            top_n = 10
        if diversity is None:
            diversity = 0.7
        if diversity > 1 or diversity < 0:
            raise ValueError("diversity should be between 0 and 1")
        if n_gram_range is None:
            n_gram_range = [(2,3)]
        if not isinstance(n_gram_range, list):
            n_gram_range = [n_gram_range]
        # check if the n_gram_range is valid
        for n_gram in n_gram_range:
            if not isinstance(n_gram, tuple):
                raise ValueError("n_gram_range should be a list of tuples")
            if len(n_gram) != 2:
                raise ValueError("n_gram_range should be a list of tuples of length 2")
            if n_gram[0] < 1 or n_gram[1] < 1:
                raise ValueError("n_gram_range should be a valid integer greater than 0")
            if n_gram[0] > n_gram[1]:
                raise ValueError("n_gram_range should start with the smaller number")

        # if model answer is emnpty then raise error
        if  len(answers[0].split()) < 1:
            # raise ValueError("there should be at least 1 answer and the model-answer")
            return zip(ids[1:], np.zeros(len(ids[1:])))
        students_keywords = self.keywords_exrtaction(answers[1:],n_gram_range,top_n,diversity)
        model_keywords = self.keywords_exrtaction(answers[0],n_gram_range,top_n,diversity)
        # pad the keywords
        model_keywords = self.pad_keywords(model_keywords, padding='post')
        if DEBUGGING:
            print(model_keywords)
        grades = self.keywords_grading(students_keywords,model_keywords)
        grades = self.keywords_grading_predict(grades)
        grades = grades.tolist()
        return zip(ids[1:], grades)
