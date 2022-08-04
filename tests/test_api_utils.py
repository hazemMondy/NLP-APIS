import utils.api_utils as api_utils
import pytest

class TestApiutilsDicttolist:
    def test_dict_to_list_happy(self):
        param1 = { "one": "Good answer", "id2v": "bad ANswer" }
        result = api_utils.dict_to_list(param1, False)
        assert result == (["one", "id2v"], ["good answer", "bad answer"]), str(result)

    def test_dict_to_list_cased(self):
        result = api_utils.dict_to_list({ "one": "Good answer", "id2v": "bad ANswer" }, True)
        assert result == (["one", "id2v"], ["Good answer", "bad ANswer"])

    def test_dict_to_list_integers(self):
        result = api_utils.dict_to_list({ 1: "good answer", 2: "bad answer" })
        assert result == (["1", "2"], ["good answer", "bad answer"])

    def test_dict_to_list_empty(self):
        result = api_utils.dict_to_list({})
        assert isinstance(result, tuple), "output is not a tuple"
        assert len(result) == 2, "output is not a tuple of length 2"
        assert result == ([], []), "output is not empty"

    def test_dict_to_list_all_floats(self):
        result = api_utils.dict_to_list({ 1: 60.0, 2.0: 55.5 })
        assert result == (["1", "2.0"], ["60.0", "55.5"]), "output is not strings"
