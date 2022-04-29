"""plagiarism_model.py"""
from typing import Optional, List, Dict#, Tuple, Union
import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import cos_sim
# from transformer_model import BERTModel

#/home/ec2-user/.local/lib/python3.7/site-packages/

# MODELPATH : str = r"models\sentence-transformers_xlm-r-distilroberta-base-paraphrase-v1"
REGRESSION_FUNCTION = np.array([0.5,0.5])


class PlagiarismModel(object):
    """PlagiarismModel"""

    def __new__(
        cls:object,
        BERTModel: object)\
            -> object:
        if not hasattr(cls, 'instance'):
            cls.instance = super(PlagiarismModel, cls).__new__(cls)
            cls.model = BERTModel.model
        return cls.instance

    # def __init__(cls, model_path=MODELPATH,
    #     # tfidf=None, regression_function=None
    #      ):
    #     cls.model = SentenceTransformer(model_path)
    #     # cls.regression_function = REGRESSION_FUNCTION
    #     # cls.tfidf = tfidf

    # @PlagiarismModel
    def __embed_corpus(
       self: object,
        corpus: List[str])\
            -> List[np.ndarray]:
        """
        embed corpus with sentence_transformers

        Args:
            corpus (list[str]): list of sentencetes strings

        Returns:
            List[np.ndarray]: embeddings of shape [(N,768)]

        """
        # type check
        if not isinstance(corpus, list):
            corpus = [corpus]

        # if passed integers
        corpus = list(map(str, corpus))

        emb = self.model.encode(corpus)

        return emb

    def __siamese_model(
       self: object,
        students_emb: List[np.ndarray])\
            -> np.ndarray:
        """
        calculate similarity between students embeddings

        Args:
            students_emb (list[np.array]): strudents embeddings of shape
                [(N,768)]

        Returns:
            np.ndarray: similarity scores of shape [(N,1,N)]

        """
        # simalrity score
        sims =np.array(
            list(
                map(
                    lambda s_emb: np.array(
                        cos_sim(
                        s_emb.reshape(1,-1),
                        students_emb))
                    , students_emb)))

        # delete self similarity
        # * can't replace
        # so we delet then insert
        # iterate over each student enumertion
        sims = np.array(
            list(
                map(
                    lambda sim:
                    np.delete(
                        sim[1],
                        obj=sim[0],
                        axis=1),
                    enumerate(
                        sims.tolist()
                    )
                )
            ))
        # insert - inf to self similarity
        # easier further on
        sims = np.array(
            list(
                map(
                    lambda sim :
                    np.insert(
                        sim[1],
                        sim[0],
                        -np.inf,
                        axis=1),
                    enumerate(
                        sims.tolist()
                    )
                )
            ))

        return sims

    def __pligarism_pipeline(
       self: object,
        students_answers: List[str],
        ids : List[int],
        threshold: Optional[float]=0.78)\
            -> List[Dict[int, float]]:
        """
        pipeline for pligarism model
            * embedding
            * siamese model
            * TF-IDF
            * regression model

        Args:
            students_answers (list[str]): students answers

        Returns:
            List[Dict[int, float]]: list of dicts of plagiarism scores
                                    cheating student_id is list index
                                    student_id:plagiarism_score is dictionary

        example:
            >>> predict([\'I am a student\',\'I am a student\',\'I am a man\'], [1456,1485,1490])
            >>> [ 1456 : { 1485 : 0.95},
                    1485 : { 1456 : 0.95},
                    1490 : {} ]

        """
        # calculate similarity with siamese model
        students_emb = self.__embed_corpus(students_answers)
        sims = self.__siamese_model(students_emb)
        # temperorary instead of regression
        res = list(
                    map(
                        lambda sim:
                            dict(
                                zip(
                                    list(
                                        map( lambda x: ids[x], # ! for correct indexing with st ids
                                            np.where(sim >= threshold)[1].tolist()
                                            )
                                        ), # student ids
                                    sim[sim >= 0.7] # similarity scores
                                    )
                                )
                        , sims
                        )
                    )

        return res

    def dummy_predict(
        self: object,
        answers: List[str])\
            -> List[Dict[int, float]]:
        """
        dummy predict for testing
        """
        return [{1:0.8,2:0.5,3:0.2},
                {4:0.8,5:0.5},answers]
    def predict(
       self: object,
        students_answers: List[str],
        ids: List[int],
        threshold: Optional[float]=0.78)\
            -> List[Dict[int, float]]:
        """
        predict plagiarism scores

        Args:
            students_answers (list[str]): students answers
            ids (list[int]): list of student ids
            threshold (Optional[float]): threshold for similarity

        Returns:
            List[Dict[int, float]]: list of dicts of plagiarism scores
                                    cheating student_id is list index
                                    student_id:plagiarism_score is dictionary

        example:
            >>> predict([\'I am a student\',\'I am a student\',\'I am a man\'], [1456,1485,1490])
            >>> [ 1456:{ 1485: 0.95}, 1485:{ 1456: 0.95} ]

        """
        scores = self.__pligarism_pipeline(students_answers,ids, threshold)
        # # return each id with coressponding scores

        res = list(
                map(
                    lambda r:
                        {r[0]: r[1]}
                    ,
                    filter(
                        lambda r:
                            any(r[1])
                        ,
                        zip(ids, scores)
                        )
                    )
                )

        return res
