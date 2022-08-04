import pytest
import utils.utils as utils

class TestUtilsFlatten(object):
    _message = lambda args: f"actual error message is: '{args[0]}' should be: '{args[1]}'"

    def test_flatten_none(self):
        actual = utils.flatten([])
        assert actual == [], self._message((actual, []))

    def test_flatten_happy(self):
        actual = utils.flatten([12, ["s0175"], 56784, [56784, 98750, "a1969"]])
        expected = [12, "s0175", 56784, 56784, 98750, "a1969"]
        assert actual == expected, self._message((actual, expected))

    def test_flatten_already_flat(self):
        actual = utils.flatten([12, "s0175", 56784, 56784, 98750, "a1969"])
        expected = [12, "s0175", 56784, 56784, 98750, "a1969"]
        assert actual == expected, self._message((actual, expected))

    def test_flatten_cubed_nested_lists(self):
        degree = None
        actual = utils.flatten([12, [["s0175"]], 56784, [56784, 98750, "a1969"]],degree)
        expected = [12, ["s0175"], 56784, 56784, 98750, "a1969"]
        assert actual == expected, self._message((actual, expected))

        degree = 3
        actual = utils.flatten([12, [["s0175"]], 56784, [56784, 98750, "a1969"]],degree)
        expected = [12, "s0175", 56784, 56784, 98750, "a1969"]
        assert actual == expected, self._message((actual, expected))

    def test_flatten_non_lists(self):
        for arg in [12, "asdfjhh", 5678.4,{"hi":123}]:
            with pytest.raises(TypeError) as exc_info:
                utils.flatten(arg)
        expected_error_msg = "input must be a list"
        message = f"actual error message is: '{exc_info.value}' should be: '{expected_error_msg}'"
        assert exc_info.match(expected_error_msg), message
