"""siamese_ner_grading_model.py"""

from typing import Optional, List, Dict, Set
import numpy as np
import spacy
from sentence_transformers.util import cos_sim
import key_words
from utils import load_obj, save_obj
from configs import configs as cfg

EXCEPTION_ENTITES = cfg['exception_entities']
NER_PATH = cfg['ner_path']
LAST_LAYER_PATH = cfg['siamese_ner_model_path']
DEBUGGING = cfg['debugging']

# load spacy model
NER = spacy.load(NER_PATH)

class SIAMESENERGradingModel(object):
    """SIAMESE_NER_GradingModel"""
    def __new__(cls: object,
        encoder_model: object) -> object:
        """
        Args:
            encoder_model: encoder_model
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super(SIAMESENERGradingModel, cls).__new__(cls)
            try:
                cls.model = encoder_model.model
            except AttributeError as err:
                if DEBUGGING:
                    print(err)
                cls.model = encoder_model
            cls.last_layer = load_obj(name = "", path = LAST_LAYER_PATH)
        return cls.instance

    def __ner(self: object,
        paragraph: str) -> Dict[str, str]:
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
        batch = 5
        for i in range(0,n_corpus,batch):
            print("Processing batch {}/{}".format(i//batch+1, (n_corpus//batch)+1))
            corpus_emb = self.model.encode(corpus[i:i+batch])
            corpus_ls.extend(corpus_emb)
        if n_corpus % batch != 0 and n_corpus > batch:
            corpus_emb = self.model.encode(corpus[i+batch:])
            corpus_ls.extend(corpus_emb)

        return np.array(corpus_ls)

    def __siamese_model(self: object,
        model_answer_emb: np.ndarray,
        students_emb: np.ndarray) -> np.ndarray:
        """
        calculate similarity between students and model answer embeddings

        Args:
            model_answer_emb (numpy.ndarray): embedding of model answer of shape
                (1,768) or 384 depending on the model
            students_emb (numpy.ndarray): embedding of students of shape
                (768, N) or 384 depending on the model

        Returns:
            np.ndarray: similarity scores of shape (N,)
 
        example:
            >>> __siamese_model(model_answer_emb, students_emb)
            [0.5, 0.5]
        """
        # type check
        if not isinstance(model_answer_emb, np.ndarray):
            model_answer_emb = np.array(model_answer_emb)

        if not isinstance(students_emb, np.ndarray):
            students_emb = np.array(students_emb)

        sims = np.array(cos_sim(model_answer_emb, students_emb))
        return sims.ravel()

    def __match_grading(self: object,
        entities: List[str],
        doc: str)-> float:
        """
        grade named entity recognition and special words in the answer
            by matching entities in doc

        Args:
            entities (list[str]): list of named entities
            doc (str): text to extract named entities from

        Returns:
            float: grade of named entity recognition

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

    def pipeline(self: object,
        docs: List[str],
        exception_entites: Optional[Set[str]]=None,
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
            >>>     exception_entites = None)
            [1]
        """
        if exception_entites is None:
            exception_entites = EXCEPTION_ENTITES

        # TODO : 1 siamese similarty
        embs = self.__embed_corpus(docs)
        model_answer_emb = embs[0]
        # embs[1:] is the rest of the embeddings for students
        sim_grades = self.__siamese_model(model_answer_emb.reshape(1,-1), embs[1:])

        # TODO : 2 named entites
        # grades = 2
        # for self.model answer only
        # clean model answer before NER
        docs[0] = key_words.clean_doc_keep_float(docs[0])
        named_entites = self.__ner(docs[0])

        ner_grades = np.zeros(len(docs[1:]))
        # length docs <= 2
        if len(docs) <= 2:
            ner_grades = np.zeros(1)

        # if there are named entites in model answer
        if named_entites:
            # grades += 1
            named_entites = list(filter(
                                    lambda x:
                                        x not in exception_entites
                                    ,named_entites
                                    ))
            # for all students answers
            # * withouth stop words removals in students answer
            ner_grades = np.array(list( map(lambda student_answer:
                    self.__match_grading(named_entites, student_answer)
                    , docs[1:])))

            if " ".join(named_entites) == docs[0]:
                return ner_grades.tolist()

        # # TODO : 3 machine learning model for wighted sum of the result
        res = self.last_layer.predict(ner_grades , sim_grades)
        return res.tolist()

    def fit(self:object,
        x_train:np.ndarray, y_train:np.ndarray,
        *args, **kwargs):
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
        """
        if path is None:
            path = LAST_LAYER_PATH
        save_obj(obj = self.last_layer, name = "", path=path)
        print("Model saved Successfully to", path)

    def predict(self: object,
        answers: List[str],
        ids: List[str],
        exception_entites: Optional[Set[str]]=None)-> List[float]:
        
        scores= self.pipeline(
                docs=answers,
                exception_entites=exception_entites)

        # ! test for length
        res = zip(ids[1:], scores)
        return res
