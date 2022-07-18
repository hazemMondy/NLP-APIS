"""grading_model.py"""
from typing import Optional, List, Dict, Tuple, Set
import spacy
import numpy as np
from sentence_transformers.util import cos_sim
import key_words


EXCEPTION_ENTITES = set(["DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL", "CARDINAL"])
# load spacy model
NER = spacy.load('en_core_web_sm')

# class GradingModel(BERTModel):
class GradingModel(object):
    """GradingModel"""

    # __metaclass__ = BERTModel

    def __new__(
        cls: object,
        BERTModel: object):
        if not hasattr(cls, 'instance'):
            cls.instance = super(GradingModel, cls).__new__(cls)
            cls.model = BERTModel.model
        return cls.instance

    def __ner(
        self: object,
        paragraph: str)\
            -> Dict[str, str]:
        """
        named entity recognition

        Args:
            paragraph (str): text to extract named entities from

        Returns:
            Dict[str, str]: named entities and their types
                key : named entity
                value : entity type
        """

        doc = NER(paragraph)
        res = {
            entity.text : entity.label_
            for entity in doc.ents
        }
        return res

    def __embed_corpus(
        self: object,
        corpus: List[str])\
            -> List[np.ndarray]:
        """
        embed list of strings sentences into list of embeddings
            using sentence_transformers model

        Args:
            corpus (List[str]): list of sentences

        Returns:
            List[numpy.ndarray]: embeddings of shape [(N,768)]
        """
        # type check
        if not isinstance(corpus, list) :
            corpus = [corpus]

        # if passed integers
        corpus = list(map(str , corpus))

        emb = self.model.encode(corpus)

        return emb


    def __siamese_model(
        self: object,
        model_answer_emb: np.ndarray,
        students_emb: np.ndarray)\
            -> np.ndarray:
        """
        calculate similarity between students and model answer embeddings

        Args:
            model_answer_emb (numpy.ndarray): embedding of model answer of shape
                (1,768)
            students_emb (numpy.ndarray): embedding of students of shape
                (768, N)

        Returns:
            np.ndarray: similarity scores of shape (N,)

        """
        # type check
        if not isinstance(model_answer_emb, np.ndarray):
            model_answer_emb = np.array(model_answer_emb)

        if not isinstance(students_emb, np.ndarray):
            students_emb = np.array(students_emb)

        sims = np.array(
            cos_sim(
            model_answer_emb, students_emb))

        return sims.ravel()

    def __match_grading(
        self: object,
        entities: List[str],
        doc: str)\
            -> float:
        """
        grade named entity recognition and special words in the answer
        by matching entities in doc

        Args:
            entities (list[str]): list of named entities
            doc (str): text to extract named entities from
        -------

        Returns:
            float: grade of named entity recognition
        -------

        example:
            >>> __match_grading(['Ahmed', 'Ali'], "Ahmed is eating food.")
            >>> 0.5

        """
        # type check
        if not isinstance(entities, list):
            entities = [entities]
        #  entities contain stop words
        grade = [True
                for entity in entities
                if entity in doc]
        try:
            return len(grade)/len(entities)
        except ZeroDivisionError:
            return -1.0

    def __emb_keywords(
        self: object,
        keywords: List[str])\
        -> List[np.ndarray]:
        """
        embed list of strings sentences into list of embeddings
            using sentence_transformers model

        Args:
            keywords (List[str]): list of sentences

        Returns:
            List[numpy.ndarray]: embeddings of shape [(N,768)]

        example:
            >>> __emb_keywords(['Ahmed', 'Ali'])
            >>> [array([[-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049],
                ...,
                [-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049],
                [-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049]]),
                array([[-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049],
                [-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049],
                ...,
                [-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049],
                [-0.049, -0.049, -0.049, ..., -0.049, -0.049, -0.049]]])]
        """
        # type check
        if not isinstance(keywords, list) :
            keywords = [keywords]

        # list[ emb (N, 768) ]
        # print(keywords)
        # print(np.array(
        #             list(
        #                 map(
        #                     self.model.encode
        #                     , keywords
        #                     )
        #                 )
        #             ).shape)
        return np.array(
                    list(
                        map(
                            self.model.encode
                            , keywords
                            )
                        )
                    )

    def pipeline(
        self: object,
        docs: List[str],
        top_n: Optional[int] = 6,
        diversity: Optional[float] = 0.5,
        n_gram_range: Optional[Tuple[int,int]] = (1,2), # (1,3)
        threshold: Optional[float] = 0.5,
        exception_entites: Optional[Set[str]]=None,
        enclosure: Optional[str]= "\"\"",
        )-> List[float]:
        """
        pipeline for grading

        Args:
            docs (List[str]): list of documents
            top_n (Optional[int]): number of top answers to return
            diversity (Optional[float]): diversity of top answers
            n_gram_range (Optional[Tuple[int,int]]): range of n-grams to consider
            threshold (Optional[float]): threshold for similarity
            exception_entites (Optional[Set[str]]): list of entities to ignore

        Returns:
            List[float]: list of grades

        example:
            >>> pipeline(
            >>>     docs = ["A student is eating food.", "A student is eating food."],
            >>>     top_n = 2,
            >>>     diversity = 0.5,
            >>>     n_gram_range = (1,2),
            >>>     threshold = 0.8,
            >>>     exception_entites = None
            >>> )
            >>> [1]
        """

        if exception_entites is None:
            exception_entites = EXCEPTION_ENTITES

        # TODO : 1 siamese similarty
        # from api not plain
        embs = self.model.encode(docs)
        model_answer_emb = embs[0]
        # embs[1:] is the rest of the embeddings for students
        sim_grades = self.__siamese_model(model_answer_emb.reshape(1,-1), embs[1:])

        # TODO : 3 named entites
        grades = 2
        # for self.model answer only
        named_entites = self.__ner(docs[0])

        # ner_grades = None
        ner_grades = np.zeros(len(docs[1:]))
        # length docs <= 2
        if len(docs) <= 2:
            ner_grades = np.zeros(1)

        # if there are named entites in model answer
        if named_entites:
            
            grades += 1
            # TODO : 2 named entites
            named_entites = list(
                                filter(
                                    lambda x:
                                        x not in exception_entites
                                    ,named_entites
                                    )
                                )
            # for all students answers
            # ! withouth stop words removals in both ner and students answer
            ner_grades = np.array(list(
                map(lambda student_answer:
                    self.__match_grading(
                        named_entites,
                        student_answer),
                    docs[1:])))

            if " ".join(named_entites) == docs[0]:
                return ner_grades.tolist()


        # if there are special in model answer
        hard_keywords = key_words.get_str_between(docs[0], enclosure)
        hard_keywords_grades = np.zeros(len(docs[1:]))
        if len(hard_keywords) != 0:
            hard_keywords_grades = key_words.hard_keywords_grading(hard_keywords, docs[1:])


        # TODO : 2 keywords
        model_candidates = key_words.candidates_tokens(docs[0],n_gram_range=n_gram_range)
        model_candidate_emb = self.model.encode(model_candidates)

        # extract keywords from model answer
        if model_answer_emb.ndim == 1:
            model_answer_emb = model_answer_emb.reshape(1, -1)

        keywords = key_words.maximal_marginal_relevance(
                model_answer_emb, model_candidate_emb,
                model_candidates,
                top_n =top_n, diversity=diversity)

        # keywords_emb = model.encode(keywords)
        students_n_grams = key_words.get_n_grams(keywords)

        students_candidates = list(map(lambda doc:
                    key_words.get_candidates(students_n_grams, doc),
                    docs[1:]))

        students_candidates_emb = list(map(self.__emb_keywords, students_candidates))
        keywords_embeddings =  list(map(self.__emb_keywords, keywords))

        keywords_grades = np.array(list(map(lambda st_cand:
                    key_words.match_keywords(keywords_embeddings, st_cand,
                    thershold=threshold),
                    students_candidates_emb,
                    )))
        res = keywords_grades + ner_grades + sim_grades
        # res = keywords_grades , ner_grades , sim_grades

        # TODO : 4 guassian regression for wighted sum of the result
        # avg sum
        res /= grades

        # TODO : 5 return the result
        return res.tolist()
        # return res

    # def fit(self, X, y=None):
    #     """
    #     fit the model

    #     Args:
    #         X (List[str]): list of documents
    #         y (None): not used

    #     Returns:
    #         object: self
    #     """
    #     return self
    def predict(
        self: object,
        answers: List[str],
        ids: List[int],
        top_n: Optional[int] = 5,
        diversity: Optional[float] = 0.7,
        n_gram_range: Optional[Tuple[int,int]] = (1,2), # (1,3)
        threshold: Optional[float] = 0.6,
        exception_entites: Optional[Set[str]]=None,
        ):
        
        scores= self.pipeline(
                docs=answers,
                top_n=top_n,
                diversity=diversity,
                n_gram_range=n_gram_range,
                threshold=threshold,
                exception_entites=exception_entites)

        # ! test for length
        res = zip(ids[1:], scores)
        return res
