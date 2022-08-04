"""manual_keywords_grading.py"""
from typing import List, Tuple, Dict, Set, Optional
import numpy as np
from sentence_transformers.util import cos_sim
from sklearn.feature_extraction.text import CountVectorizer
import utils.key_words as key_words
from utils.utils import flatten, KeywordType
from utils.strings_utils import get_str_between, clean_doc
from configs.configs import configs as cfg

DEBUGGING = cfg['debugging']
MEDIUM_THRESHOLD = cfg['threshold_medium']
SOFT_THRESHOLD = cfg['threshold_soft']
MEDIUM_ENCLOSURE = cfg['enclosure_medium']
SOFT_ENCLOSURE = cfg['enclosure_soft']
HARD_ENCLOSURE = cfg['enclosure_hard']
DEFAULT_WEIGHT = cfg['default_weight']
DEFAULT_EMPTY_WEIGHT = cfg['default_empty_weight']

if not DEBUGGING:
    import warnings
    warnings.filterwarnings('ignore')
class ManualKeywordsGradingModel(object):
    """ManualKeywordsGradingModel"""
    # words_emb_dict = {}

    def __new__(cls: object,encoder_model: object):
        """
        Args:
            encoder_model: encoder_model
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(ManualKeywordsGradingModel, cls).__new__(cls)
            try:
                cls.model = encoder_model.model
            except AttributeError as err:
                if DEBUGGING:
                    print(err)
                cls.model = encoder_model
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
            >>> mkgm = KeywordsGradingModel(BERTModel)
            >>> mkgm._get_word_emb('hello')
            array([0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97])
        """
        if word in self.words_emb_dict:
            return self.words_emb_dict[word]
        else:
            self.words_emb_dict[word] = self.model.encode(word)
            return self.words_emb_dict[word]

    def __candidates_tokens(self:object,doc:str,
        n_gram_range : Optional[List[Tuple[int,int]]]= [(2,3)])-> List[str]:
        """
        extract candidates from a document

        Args:
            doc (str): document
            n_gram_range (Optional[Tuple[int,int]]): n_gram range

        Returns:
            List[str]: list of candidates words/phrases

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> doc = 'You need to know how much vinegar was used in each container.'
            >>> mkgm.__candidates_tokens(doc)
            ['know vinegar used', 'vinegar used container', 'need know vinegar']
        """
        try:
            count = CountVectorizer(
                ngram_range=n_gram_range).fit([doc])
            candidates = count.get_feature_names()
        except ValueError as err:
            if "empty vocabulary" in str(err):
                if DEBUGGING:
                    print("empty vocabulary")
            return [*set(doc.split())]
        return [*set(candidates)]

    def _get_candidates(self:object,n_grams:List[Tuple[int,int]], doc:str)->List[str]:
        """
        Returns the candidates for the given n-grams.

        Args:
            n_grams: List of n-grams.
            doc: The document.

        Returns:
            List of candidates.

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> n_grams = [(1,2),(2,3)]
            >>> doc = 'You need to know how much vinegar was used in each container.'
            >>> mkgm._get_candidates(n_grams,doc)
            ['know vinegar used', 'vinegar used container', 'need know vinegar']
        """
        return flatten(list(map(lambda gram :
            self.__candidates_tokens(str(doc), n_gram_range=gram), n_grams)))

    def _emb_keywords(self:object,keywords:List[str])->np.ndarray:
        """
        Args:
            keywords: List of keywords.
            doc: The document.

        Returns:
            np.array of shape (docs_number,768) or (docs_number,384) depending on the encoder model

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> keywords = ['hello','world']
            >>> mkgm.emb_keywords_(keywords)
            array([[0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97],
                   [0.01, 0.02, 0.03, ..., 0.99, 0.98, 0.97]])
        """
        return np.array(list(map(self._get_word_emb, keywords)))

    def keywords_exrtaction(self:object,
        docs:List[str], keywords:Set[str], batch:int = 5,
        *args,**kwargs):
        """
        keywords exrtaction pipeline

        Args:
            docs (List[str]): list of documents
            keywords (Set[str]): set of keywords
            batch (Optional[int]): batch size

        Returns:
            List[List[str]]: list of top_n keyprhases

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> docs = ['You need to know how much vinegar was used in each container.']
            >>> keywords = {'vinegar','container'}
            >>> mkgm.keywords_exrtaction(docs,keywords)
            array([[0.789, 0.987, 0.32484 ....]])  (not real example)
        """
        if isinstance(docs, str):
            docs = [docs]
        n_docs = len(docs)
        if n_docs < batch:
            batch = n_docs

        ngrams = [*set(list(map(lambda x : len(x.split(' ')),  list(keywords))))]
        max_n = max(ngrams)+1
        ngrams = min(ngrams),max_n
        ngrams = [ngrams]
        # print(ngram)
        # all model answers
        docs_keys_emb_ls = []
        # do in batches
        if DEBUGGING:
            print([i for i in range(0,n_docs,batch)])
            print(n_docs)
        for i in range(0,n_docs,batch):
            if DEBUGGING:
                print("Processing batch {}/{}".format(i//batch+1, (n_docs//batch)+1))
            docs_candidates = list(map(lambda doc: self._get_candidates(ngrams, doc),
                docs[i:i+batch]))
            if DEBUGGING:
                print("Extracting embeddings")
            docs_candidates_emb = list(map(self._emb_keywords,docs_candidates))
            docs_keys_emb_ls.extend(docs_candidates_emb)
        if n_docs % batch != 0 and n_docs > batch:
            docs_candidates = list(map(lambda doc: self._get_candidates(ngrams, doc),
                docs[i+batch:]))
            docs_candidates_emb =  list(map(self._emb_keywords,docs_candidates))
            docs_keys_emb_ls.extend(docs_candidates_emb)
        return docs_keys_emb_ls

    def keywords_grading(self:object,
        answers_candidates_emb: List[List[str]], answers:List[str], manual_keywords:List[List[str]],
        grading_type:str = "soft" or "medium" or "hard")-> np.ndarray:
        """
        Args:
            docs_keywords (List[List[str]]): list of top_n keyprhases for each document
            model_answer_keywords (List[List[str]]): list of top_n keyprhases for each model answer

        Returns:
            np.array of shape (docs_number,top_n) containing the scores for
                each keyword for each document

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> docs_keywords = [['know vinegar used', 'vinegar used container',
                 'need know vinegar']]
            >>> model_answer_keywords = [['know vinegar used', 'vinegar used container',
                 'need know vinegar']]
            >>> mkgm.keywords_grading(docs_keywords,model_answer_keywords)
            array([[[1.0000002 , 1.0000002 , 0.99999964]]], dtype=float32)
        """
        if grading_type.lower() == "hard":
            grades = key_words.hard_keywords_grading(manual_keywords, answers)\
                .astype(float).round(3).clip(0,1)
            return grades.T
        if grading_type.lower() == "soft":
            threshold = SOFT_THRESHOLD
        if grading_type.lower() == "medium":
            threshold = MEDIUM_THRESHOLD

        embs = self.model.encode(manual_keywords)
        if DEBUGGING:
            list(map(lambda docs_cand :print(("before",embs.shape, docs_cand.shape))
                ,answers_candidates_emb))
            np.array(list(map(lambda docs_cand :
            print("after",(cos_sim(embs, docs_cand).__array__().max(axis=1) > threshold).shape),
                        answers_candidates_emb))).astype(float).round(3).clip(0,1)

        out = np.array(list(map(lambda docs_cand :cos_sim
                    (embs, docs_cand).
                    __array__().max(axis=1) > threshold,
                    answers_candidates_emb))).astype(float).round(3).clip(0,1)
        if DEBUGGING:
            print("grade shape", out.T.shape)
        return out.T

    def keywords_grading_predict(self:object, hard_grades:np.ndarray,
        medium_grades:np.ndarray, soft_grades:np.ndarray,
        hard_weights:List[float], medium_weights:List[float], soft_weights:List[float],
        n_essays:int)->np.ndarray:
        """
        Args:
            hard_grades (np.ndarray): hard grades for each document
            medium_grades (np.ndarray): medium grades for each document
            soft_grades (np.ndarray): soft grades for each document
            hard_weights (List[float]): weights for hard grading
            medium_weights (List[float]): weights for medium grading
            soft_weights (List[float]): weights for soft grading
            n_essays (int): number of essays to grade

        Returns:
            np.array of shape (docs_number,) containing the final scores

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> grades = np.array([[[0.761, 0.242, 0.263, 0.832, 0.437,
                0.259, 0.269, 0.314, 0.218,0.237],
                [0.509, 0.392, 0.472, 0.451, 0.924, 0.366, 0.24 , 0.314, 0.236,
                0.237]]], dtype=float32)
            >>> mkgm.keywords_grading_predict(grades)
            array([[0.467, 0.401]])
        """
        weights_dict = { "hard": hard_weights, "medium": medium_weights, "soft": soft_weights }
        grades_dict = { "hard": hard_grades, "medium": medium_grades, "soft": soft_grades }
        weights, grades = 0, 0
        for keyword_type, grade_weight in weights_dict.items():
            if grade_weight[0] == DEFAULT_EMPTY_WEIGHT:
                continue
            elif len(grade_weight) == 0:
                grade_weight = np.array([DEFAULT_WEIGHT])
            elif grade_weight[0] == " ":
                grade_weight = np.array([DEFAULT_WEIGHT])
            if isinstance(weights, int):
                weights = np.array(grade_weight)
                grades = grades_dict[keyword_type]
            else:
                weights = np.concatenate((weights, grade_weight))
                grades = np.concatenate((grades, grades_dict[keyword_type]))

        if DEBUGGING:
            for _, grade in grades_dict.items():
                print("shape", grade.shape)
            print("grades shape", grades.shape)
            print("weights shape", weights.shape)
            print("weights", weights)

        if isinstance(weights, int):
            if DEBUGGING:
                print("no keywords available")
                print("weights", weights)
            return np.zeros((n_essays,)).ravel()

        # fill weights nan with mean
        # get weights > 0 to ignore empty weights
        means = weights[weights > DEFAULT_WEIGHT]
        if means.size == 0:
            means = np.array([1])
        weights = np.nan_to_num(weights, nan=np.nanmean(means))

        if DEBUGGING:
            print("weights", weights)
        if weights.shape[0] == 0:
            if DEBUGGING:
                print(grades.shape)
            return grades.mean(axis=0)
        # * instructor forgot to add weights to the grades
        if not weights.any():
            if DEBUGGING:
                print("weights all 0", weights, grades.mean(axis=0).shape, grades.shape)
            grades = 0
            for _, grade in grades_dict.items():
                if grade.any():
                    if isinstance(grades,int):
                        if DEBUGGING:
                            print("Grade is 0")
                        grades = grade
                    else:
                        grades = np.concatenate((grades, grade))
            return grades.mean(axis=0)

        return np.average(grades, axis=0, weights=weights).astype(float).round(3).clip(0,1)

    def pipeline(self:object, answers:List[str], enclosure:str,
        grading_type:str = "soft" or "medium" or "hard")->Tuple[str, np.ndarray, List[float]]:
        """
        the pipeline processs done for each keyword type gradeing

        Args:
            answers (List[str]): list of model answers
            enclosure (str): enclosure of the keywords
            grading_type (str): grading type of the keywords

        Returns:
            str: the model answer cleaned from the keywords and the enclosure and
                the weights (if any)
            np.array of shape (n_model_answers,docs_number) containing
                the scores for each keyword for each document
            List[float]: list of weights for each keyword

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> answers = ["know \"\"vinegar\"\"0.8 used", "vinegar used container",
                "need know the size"]
            >>> enclosure = "\"\""
            >>> grading_type = "hard"
            >>> mkgm.pipeline(answers, enclosure, grading_type)
            ('know used', array([1.0 , 1.0 , 0.0]), [DEFAULT_WEIGHT])
        """
        keywords = get_str_between(answers[0], enclosure)
        if DEBUGGING:
            print(keywords)
        grades = np.zeros(len(answers[1:]))
        keywords_weights = [DEFAULT_EMPTY_WEIGHT]
        if len(keywords) != 0:
            if DEBUGGING:
                print("keywords:", keywords)
            students_keywords = self.keywords_exrtaction(docs = answers[1:], keywords=keywords)
            keywords_weights = key_words.get_weights_from_doc(answers[0], keywords, enclosure)
            if DEBUGGING:
                print("keywords_weights:", keywords_weights)
            grades = self.keywords_grading(answers_candidates_emb = students_keywords,
                manual_keywords = keywords, answers = answers[1:], grading_type = grading_type)
            # clean model answer
            answers[0] = clean_doc(answers[0], keywords, keywords_weights, enclosure)

        return answers[0], grades, keywords_weights

    def predict(
        self: object,
        answers: List[str],
        ids: List[str])-> Dict[str, float]:
        """
        the main function to process the answers and return the scores
        Args:
            answers (List[str]): list of answers
            ids (List[str]): list of ids of the corresponding answers

        Returns:
            Dict[str, float]: dictionary of ids with their scores

        example:
            >>> mkgm = ManualKeywordsGradingModel(BERTModel)
            >>> answers = ["know \"\"vinegar\"\"0.8 used", "vinegar used container",
                "need know the size"]
            >>> ids = list(range(len(answers)))
            >>> mkgm.predict(answers,ids)
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

        answers[0], keywords_medium_grades, keywords_medium_weights = self.pipeline(
            answers, MEDIUM_ENCLOSURE, KeywordType.MEDIUM.__repr__())
        answers[0], keywords_soft_grades , keywords_soft_weights = self.pipeline(
            answers, SOFT_ENCLOSURE, KeywordType.SOFT.__repr__())
        answers[0], keywords_hard_grades , keywords_hard_weights = self.pipeline(
            answers, HARD_ENCLOSURE, KeywordType.HARD.__repr__())
        # grading
        grades = self.keywords_grading_predict(
            hard_grades = keywords_hard_grades, medium_grades = keywords_medium_grades,
            soft_grades = keywords_soft_grades, hard_weights = keywords_hard_weights,
            medium_weights = keywords_medium_weights, soft_weights = keywords_soft_weights,
            n_essays=len(answers[1:]))
        return zip(ids[1:], grades)
