import pytest
import numpy as np
import utils.key_words as key_words

class TestKeywordsGetweightsfromdoc(object):
    def test_get_weights_from_doc_happy_uncompeleted_floats(self):
        doc = "hi my name is \"\"john\"\"4. and I am a student"
        actual = key_words.get_weights_from_doc(doc, ["john"], "\"\"")
        expected = [4.0]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message

    def test_get_weights_from_doc_happy_2(self):
        doc = "hi my \"\"@name@\"\"2 is \"\"@john@\"\"4 and I am a student"
        actual = key_words.get_weights_from_doc(doc, ["name", "john"], "\"\"@")
        expected = [2.0, 4.0]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message

    def test_get_weights_from_doc_empty_weights(self):
        doc = "hi my name is \"\"john\"\" and I am a student"
        actual = key_words.get_weights_from_doc(doc, ["john"], "\"\"")
        expected = [key_words.DEFAULT_WEIGHT]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message

    def test_get_weights_from_doc_mixed_keywords(self):
        doc = "hi my \"\"@name@\"\"2 is \"\"john\"\"4 and I am a \"\"@student\"\"@"
        actual = key_words.get_weights_from_doc(doc, ["name","student"], "\"\"@")
        expected = [2.0, key_words.DEFAULT_WEIGHT]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message

    def test_get_weights_from_doc_nones(self):
        # keywords is None
        doc = "hi my name is \"\"john\"\"4 and I am a student"
        enclosure = "\"\""
        keywords = ["john"]
        with pytest.raises(ValueError) as exc_info:
            key_words.get_weights_from_doc(doc, None, enclosure)
        expected_error_msg = "keywords is None"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message
        # doc is None
        with pytest.raises(ValueError) as exc_info:
            key_words.get_weights_from_doc(None, keywords, enclosure)
        expected_error_msg = "doc is None"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message
        # enclosure is none
        with pytest.raises(ValueError) as exc_info:
            key_words.get_weights_from_doc(doc, keywords, None)
        expected_error_msg = "enclosure is None"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

    @pytest.mark.xfail
    def test_get_weights_from_doc_non_str_doc(self):
        doc = 1
        keywords = ["john"]
        enclosure = "\"\""
        actual = key_words.get_weights_from_doc(doc, keywords, enclosure)
        expected = [key_words.DEFAULT_WEIGHT]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message
    @pytest.mark.xfail
    def test_get_weights_from_doc_non_str_doc_2(self):
        doc = [1.0]
        keywords = ["john"]
        enclosure = "\"\""
        actual = key_words.get_weights_from_doc(doc, keywords, enclosure)
        expected = [key_words.DEFAULT_WEIGHT]
        message = f"actual is: '{actual}', expected: '{expected}'"
        assert actual == expected, message

class TestKeywordsCandidatestokens(object):
    def test_candidates_tokens_none_doc(self):
        actual = key_words.candidates_tokens(None)
        expected = []
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert actual == expected, message

    def test_candidates_tokens_empty_doc(self):
        actual = key_words.candidates_tokens("")
        expected = []
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert actual == expected, message

    def test_candidates_tokens_doc_happy(self):
        arg = "ahmed go to the store and buy some milk"
        actual = key_words.candidates_tokens(arg)
        # assert actual is list of strings
        message = lambda args: f"actual type is: '{type(args[0])}' \
            type should be: '{type(args[1])}'"
        assert isinstance(actual, list), message([actual, []])
        for out in actual:
            assert isinstance(out, str), message([out, ""])

    def test_candidates_tokens_none_ngrams(self):
        arg = "ahmed go to the store and buy some milk"
        actual = key_words.candidates_tokens(arg)
        # assert actual is list of strings
        message = lambda args: f"actual type is: '{type(args[0])}' \
            type should be: '{type(args[1])}'"
        assert isinstance(actual, list), message([actual, []])
        assert len(actual) > 0, "should be more than 0"
        for out in actual:
            assert isinstance(out, str), message([out, ""])

class TestKeywordsGetngrams(object):
    def test_get_n_grams_happy(self):
        result = key_words.get_n_grams(["yes", "go ahead"])
        assert result == [(1, 2), (2, 3)]

    def test_get_n_grams_same_ngrams(self):
        result = key_words.get_n_grams(["yes", "no"])
        assert result == [(1, 2)]

    def test_get_n_grams_empty_keywords(self):
        result = key_words.get_n_grams([])
        assert result == []

    def test_get_n_grams_none_keywords(self):
        with pytest.raises(ValueError) as exc_info:
            key_words.get_n_grams(None)
        expected_error_msg = "keywords is None"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

class TestKeywordsGetcandidates(object):
    def test_get_candidates_1(self):
        doc = """The three most well known thermoset compounds are \
            Urea-formaldehyde, urethane and epoxy compounds.Thermoset \
            plastics, as the name suggests, are materials that set or \
            harden with heat exposure. The thermosetting plastic compounds \
            are mostly used in buildings and other construction projects \
            where they can be molded into intricate shapes. Advantages of \
            using thermosetting plastics. This type of substance has several\
            advantages when it comes to industrial design because of its strength \
            and durability. It cannot be melted down again because it has been \
            converted into a hardened substance. The thermoset is also non-flammable \
            and quite affordable, about $600 per cubic meter. This means that it is \
            very durable and lasts for a long period of time."""
        actual = key_words.get_candidates([(1, 2), (2, 3)], doc)
        message = lambda args: f"actual type is: '{type(args[0])}' \
            type should be: '{type(args[1])}'"
        assert isinstance(actual, list), message([actual, []])
        for out in actual[:5]:
            assert isinstance(out[0], str), message([out[0], ""])

    def test_get_candidates_none_doc(self):
        ngrams = [(1, 2), (2, 3)]
        actual = key_words.get_candidates(ngrams, None)
        assert len(actual) == len(ngrams), "should be equal length"

    def test_get_candidates_empty_doc(self):
        actual = key_words.get_candidates([(1, 2), (2, 3)], "")
        message = lambda args: f"actual type is: '{type(args[0])}' \
            type should be: '{type(args[1])}'"
        assert isinstance(actual, list), message([actual, []])
        print(actual)
        for out in actual:
            assert len(out)==0, "length of sub-lists should be 0"

class TestKeywordsHardkeywordsgrading(object):
    def test_hard_keywords_grading_happy(self):
        docs = ["people are awesome", "theory", "science"]
        keywords = ["people", "theory"]
        actual = key_words.hard_keywords_grading(keywords, docs)
        expected = np.array([[1., 0],[0, 1],[0, 0]])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.allclose(actual, expected), message

    def test_hard_keywords_grading_none_list_docs(self):
        docs = "people are awesome"
        keywords = ["people", "theory"]
        actual = key_words.hard_keywords_grading(keywords, docs)
        expected = np.array([[1.0, 0]])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.allclose(actual, expected), message

    def test_hard_keywords_grading_non_list_keywords(self):
        docs = ["people are awesome", "theory", "science"]
        keywords = "people"
        actual = key_words.hard_keywords_grading(keywords, docs)
        expected = np.array([[1.0], [0], [0]])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.allclose(actual, expected), message

    def test_hard_keywords_grading_empty_list_docs(self):
        docs = []
        keywords = ["people", "theory"]
        actual = key_words.hard_keywords_grading(keywords, docs)
        expected = np.array([[0, 0]])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.allclose(actual, expected), message

    def test_hard_keywords_grading_none_docs(self):
        docs = None
        keywords = ["people", "theory"]
        actual = key_words.hard_keywords_grading(keywords, docs)
        expected = np.array([[0, 0]])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.allclose(actual, expected), message

    def test_hard_keywords_grading_empty_list_keywords(self):
        docs = ["people are awesome", "theory", "science"]
        keywords = []
        with pytest.raises(ValueError) as exc_info:
            key_words.hard_keywords_grading(keywords, docs)
        expected_error_msg = "keywords is empty"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

    def test_hard_keywords_grading_none_keywords(self):
        docs = ["people are awesome", "theory", "science"]
        keywords = None
        with pytest.raises(ValueError) as exc_info:
            key_words.hard_keywords_grading(keywords, docs)
        expected_error_msg = "keywords is None"
        message = f"actual error msg is: '{str(exc_info.value)}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message
