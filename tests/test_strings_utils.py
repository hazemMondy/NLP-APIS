import pytest
import numpy as np
import utils.strings_utils as strings_utils

class TestStringsutilsReversestring(object):
    def test_reverse_string_happy(self):
        arg = """SELECT * FROM Movies WHERE Title=\'Jurassic Park\'"""
        actual = strings_utils.reverse_string(arg)
        expected = """\'kraP cissaruJ\'=eltiT EREHW seivoM MORF * TCELES"""
        assert actual == expected, "actual actual is: '{}' should be: '{}'"\
            .format(actual, expected)

    def test_reverse_string_empty(self):
        actual = strings_utils.reverse_string("")
        assert actual == "", "empty string"

    def test_reverse_string_none(self):
        actual = strings_utils.reverse_string(None)
        assert actual is None, "actual is not None"

    def test_reverse_string_non_strings(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.reverse_string(1.2)
        expected_error_msg = "doc must be a string"
        assert exc_info.match(expected_error_msg),\
             "actual error message is: '{}' should be: '{}'".\
                format(exc_info, expected_error_msg)

    def test_reverse_string_non_strings_2(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.reverse_string([1])
        expected_error_msg = "doc must be a string"
        assert exc_info.match(expected_error_msg),\
             "actual error message is: '{}' should be: '{}'".\
                format(exc_info, expected_error_msg)

class TestStringsutilsGetstrbetween(object):
    def test_get_str_between_happy(self):
        actual = strings_utils.get_str_between("DROP \"\"TABLE\"\"tmp;", "\"\"")
        assert actual == ["TABLE"], "actual output is: '{}' should be: '{}'"\
            .format(actual, ["TABLE"])

    def test_get_str_between_not_found(self):
        actual = strings_utils.get_str_between("DROP TABLE tmp;", "\"\"")
        assert actual == []

    def test_get_str_between_empty_doc(self):
        actual =  strings_utils.get_str_between("", "\"\"")
        assert actual == [], "actual output is: '{}' should be: '{}'"\
            .format(actual, [])

    def test_get_str_between_none_doc(self):
        actual =  strings_utils.get_str_between(None, "\"\"")
        assert actual == [], "actual output is: '{}' should be: '{}'"\
            .format(actual, [])

    def test_get_str_between_none_enclosure(self):
        actual = strings_utils.get_str_between("DROP \"\"TABLE\"\"tmp;", None)
        assert actual == ["TABLE"], "actual output is: '{}' should be: '{}'"\
            .format(actual, ["TABLE"])

    def test_get_str_between_non_sting_enclosure(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.get_str_between("DROP \"\"TABLE\"\"tmp;", 123)
        expected_error_msg = "enclosure must be a string"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)

    def test_get_str_between_invalid_enclosure(self):
        with pytest.raises(ValueError) as exc_info:
            strings_utils.get_str_between("DROP \"\"TABLE\"\"tmp;", "")
        expected_error_msg = "enclosure must be a valid non-empty string"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)

    def test_get_str_between_non_string_doc(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.get_str_between(123, "\"\"")
        expected_error_msg = "doc must be a string"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)

    def test_get_str_between_non_string_inside_keywords(self):
        actual = strings_utils.get_str_between("DROP \"\"1231\"\"tmp;", "\"\"")
        assert actual == ["1231"], "actual output is: '{}' should be: '{}'"\
            .format(actual, ["1231"])


class TestStringsutilsCleandoc(object):
    def test_clean_doc_happy(self):
        actual = strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
            ["TABLE"], [1.0], "\"\"")
        assert actual == "DROP tmp;"

    def test_clean_doc_normal(self):
        doc = "\"\"good\"\" \"\"name\"\"2 \"\"increase\"\"  bad cocaine"
        actual = strings_utils.clean_doc(doc, ["good","name","increase"],
            [np.nan, 2.0, np.nan], "\"\"")
        assert actual == " bad cocaine", "actual is: '{}' should be: '{}'"\
            .format(actual, " bad cocaine")

    # ! will retrun the doc with the same invalid float value
    def test_clean_doc_uncompleted_floats(self):
        actual = strings_utils.clean_doc("DROP \"\"TABLE\"\"1. tmp;",
            ["TABLE"], [1.0], "\"\"")
        assert actual == "DROP tmp;", "actual is: '{}' should be: '{}'"\
            .format(actual, "DROP tmp;")

    def test_clean_doc_non_list_keywords(self):
        actual = strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
            "TABLE", [1.0], "\"\"")
        assert actual == "DROP tmp;"

    def test_clean_doc_none_doc(self):
        actual = strings_utils.clean_doc(None, ["TABLE"], [1.0], "\"\"")
        assert actual == None

    def test_clean_doc_none_keywords(self):
        actual = strings_utils.clean_doc("DROP TABLE tmp;", None, [1.0], "\"\"")
        assert actual == "DROP TABLE tmp;"


    def test_clean_doc_none_weights(self):
        with pytest.raises(ValueError) as exc_info:
            strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
                ["TABLE"], None, "\"\"")
        expected_error_msg = "weights must be a list of floats"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)

    def test_clean_doc_none_enclusre(self):
        actual = strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
            ["TABLE"], [1.0], None)
        assert actual == "DROP tmp;"

    def test_clean_doc_non_sting_enclosure(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
                ["TABLE"], [1.0], 123)
        expected_error_msg = "enclosure must be a string"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)

    def test_clean_doc_invalid_enclosure(self):
        with pytest.raises(ValueError) as exc_info:
            strings_utils.clean_doc("DROP \"\"TABLE\"\"1.0 tmp;",
                ["TABLE"], [1.0], "")
        expected_error_msg = "enclosure must be a valid non-empty string"
        assert exc_info.match(expected_error_msg),\
                "actual error message is: '{}' should be: '{}'".\
                    format(exc_info, expected_error_msg)


class TestStringsutilsCleanpunctuation(object):
    def test_clean_punctuation_none(self):
        actual = strings_utils.clean_punctuation(None)
        assert actual is None

    def test_clean_punctuation_happy(self):
        actual = strings_utils.clean_punctuation("i go to school.?!")
        assert actual == "i go to school"

    def test_clean_punctuation_non_strings(self):
        actual = strings_utils.clean_punctuation(1452)
        assert actual == "1452"

    def test_clean_punctuation_spicial_chars(self):
        # ! not supported yet
        # Exceptions = ['#','$','%','^','&','*'
        #     ,'(',')','_','+','-','=','[',']'
        #     ,'|',';','\'',':',',','/','<','>']
        doc= "8º #$%^&*()_+-=[ ] { }  | ; ' : ,  . / < > "
        out= "8º #$%^&*()_+-=[ ] | ; ' : , / < > "
        actual = strings_utils.clean_punctuation(doc)
        assert actual == out, f"actual is: '{actual}' should be: '{out}'"

class TestStringsutilsParsefloat(object):
    def test_parse_float_none(self):
        actual = strings_utils.parse_float(None)
        assert actual == []

    def test_parse_float_happy(self):
        actual = strings_utils.parse_float("so hogh 1.2")
        assert actual == [1.2]

    # *expected to fail as it only check for float values
    @pytest.mark.xfail()
    def test_parse_float_integers_in_doc(self):
        actual = strings_utils.parse_float("so hogh 1")
        assert actual == [1.0]

    # *expected to fail as it only check for float values
    @pytest.mark.xfail()
    def test_parse_float_not_completed_floats(self):
        actual = strings_utils.parse_float("so hogh 2. ")
        assert actual == [2.0]

    # *expected to fail as it only check for float values
    @pytest.mark.xfail()
    def test_parse_float_non_strings_integers(self):
        arg = 123
        # test on type error
        with pytest.raises(TypeError) as exc_info:
            strings_utils.parse_float(arg)
        expected_error_msg = "doc must be a string"
        message = f"actual error message is: '{exc_info}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

        # assert output
        actual = strings_utils.parse_float(str(arg))
        message = f"actual is: '{actual}' should be: '{arg}'"
        assert actual == [float(str(arg))], message

    def test_parse_float_non_strings_floats(self):
        arg = 123.0
        # test on type error
        with pytest.raises(TypeError) as exc_info:
            strings_utils.parse_float(arg)
        expected_error_msg = "doc must be a string"
        message = f"actual error message is: '{exc_info}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

        # assert output
        actual = strings_utils.parse_float(str(arg))
        message = f"actual is: '{actual}' should be: '{arg}'"
        assert actual == [float(str(arg))], message


class TestStringsutilsCleandockeepfloat(object):
    def test_clean_doc_keep_float_normal(self):
        actual = strings_utils.clean_doc_keep_float("is1.0 distroying. 2.0 humans3 life. ?")
        assert actual == "is1.0 distroying 2.0 humans3 life "

    def test_clean_doc_keep_float_non_strings_integers(self):
        actual = strings_utils.clean_doc_keep_float(789.4)
        assert actual == "789.4"

    def test_clean_doc_keep_float_non_strings_none_int_float_str(self):
        with pytest.raises(TypeError) as exc_info:
            strings_utils.clean_doc_keep_float(["asd"])
        expected_error_msg = "doc must be a string"
        message = f"actual error message is: '{exc_info}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

    def test_clean_doc_keep_float_none(self):
        actual = strings_utils.clean_doc_keep_float(None)
        # assert actual == ""
        assert actual is None


class TestStringsutilsMatchkeywordsindoc(object):
    def test_match_keywords_in_doc_happy(self):
        args = ["people", "theory"], "people of africa"
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([1,0])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

    def test_match_keywords_in_doc_none_doc_and_empty_none_string(self):
        args = ["people", "theory"], None
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([0,0])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

        # test on empty string
        args = ["people", "theory"], ""
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([0,0])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

        # test on int for type error
        args = ["people", "theory"], 123
        with pytest.raises(TypeError) as exc_info:
            strings_utils.match_keywords_in_doc(*args)
        expected_error_msg = "doc must be a string"
        message = f"actual error message is: '{exc_info}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

    def test_match_keywords_in_doc_none_keywords_and_empty(self):
        args = None, "people of africa"
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([0,])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

        args = [], "people of africa"
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([0,])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

    def test_match_keywords_in_doc_str_keywords(self):
        args = "people", "people of africa"
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([1,])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message

    def test_match_keywords_in_doc_non_str_keywords_elements(self):
        args = [123], "people of africa"
        with pytest.raises(TypeError) as exc_info:
            strings_utils.match_keywords_in_doc(*args)
        expected_error_msg = "keywords must be a list of strings"
        message = f"actual error message is: '{exc_info}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message

        args = ["123"], "people of africa"
        actual = strings_utils.match_keywords_in_doc(*args)
        expected = np.array([0,])
        message = f"actual is: '{actual}' should be: '{expected}'"
        assert np.array_equal(actual, expected), message
