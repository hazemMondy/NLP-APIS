"""plagiarism_model.py"""
from typing import Optional, List, Dict#, Tuple, Union
import numpy as np
from sentence_transformers.util import cos_sim
from configs import configs as cfg

DEBUGGING = cfg['debugging']
if not DEBUGGING:
    import warnings
    warnings.filterwarnings('ignore')
class PlagiarismModel(object):
    """PlagiarismModel"""

    def __new__(cls:object,
        encoder_model: object) -> object:
        if not hasattr(cls, 'instance'):
            cls.instance = super(PlagiarismModel, cls).__new__(cls)
            try:
                cls.model = encoder_model.model
            except AttributeError as err:
                if DEBUGGING:
                    print(err)
                cls.model = encoder_model
        return cls.instance

    def __embed_corpus(self: object,
        corpus: List[str], batch:int = 50)-> List[np.ndarray]:
        """
        embed list of strings sentences into list of embeddings
            using sentence_transformers model

        Args:
            corpus (List[str]): list of sentences

        Returns:
            List[numpy.ndarray]: embeddings of shape [(N,768)] or 384 depending on the model

        example:
            >>> __siamese_model(["how much vinegar" ,"used in each container"])
            array([[ 9.69761238e-02,  1.09762438e-01, -1.33646965e-01,
                -5.82718849e-02,  1.10034369e-01, -1.03692561e-02,
                2.60011166e-01,  2.99603552e-01,  7.14241946e-03,
                ...]
                [5.96261024e-01, -6.62316903e-02, -3.36377978e-01,
                1.44604310e-01,  5.11792123e-01,  2.44314805e-01,
                ...]], dtype=float32)
        """
        # type check
        if not isinstance(corpus, list) :
            corpus = [corpus]

        # if passed integers
        corpus = list(map(str , corpus))
        corpus_ls = []
        # do in batches
        n_corpus = len(corpus)
        for i in range(0,n_corpus,batch):
            if DEBUGGING:
                print("Processing batch {}/{}".format(i//batch+1, (n_corpus//batch)+1))
            corpus_emb = self.model.encode(corpus[i:i+batch])
            corpus_ls.extend(corpus_emb)
        if n_corpus % batch != 0 and n_corpus > batch:
            corpus_emb = self.model.encode(corpus[i+batch:])
            corpus_ls.extend(corpus_emb)

        return np.array(corpus_ls)

    def __siamese_model(self: object,
        students_emb: List[np.ndarray]) -> np.ndarray:
        """
        calculate similarity between students embeddings

        Args:
            students_emb (list[np.array]): strudents embeddings of shape
                [(N,768)]

        Returns:
            np.ndarray: similarity scores of shape [(N,1,N)]

        """
        # simalrity score
        sims =np.array(list(map(
                    lambda s_emb: np.array(
                        cos_sim( s_emb.reshape(1,-1),students_emb))
                    , students_emb)))

        # delete self similarity
        # * can't replace
        # so we delet then insert
        # iterate over each student enumertion
        sims = np.array(list(map(
                    lambda sim:
                    np.delete(
                        sim[1],
                        obj=sim[0],
                        axis=1),
                    enumerate(sims.tolist())
                )))
        # insert - inf to self similarity
        # easier further on
        sims = np.array(list(map(
                    lambda sim :
                    np.insert(
                        sim[1],
                        sim[0],
                        -np.inf,
                        axis=1),
                    enumerate(sims.tolist())
                )))

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
        res = list(map(lambda sim:
                    dict(zip(list(
                                map( lambda x: ids[x], # ! for correct indexing with st ids
                                    np.where(sim >= threshold)[1].tolist())), # student ids
                            sim[sim >= threshold] # similarity scores
                            )), sims))
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

        res = list(map(lambda r:
                        {r[0]: r[1]},
                    filter(lambda r:
                            any(r[1]),
                        zip(ids, scores)
                        )))
        return res
