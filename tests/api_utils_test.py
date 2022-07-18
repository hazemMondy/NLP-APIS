import api_utils

class Test_Api_utils_Dict_to_list:
    def test_dict_to_list_1(self):
        result = api_utils.dict_to_list({ "Synch": "Synchronised", "45": "Synchron" })
        assert result == (["synch", "45"], ["synchronised", "synchron"])

    def test_dict_to_list_2(self):
        result = api_utils.dict_to_list({ "": 123 })
        assert result == ([""], ["123"])
