from smplcodec.utils import to_camel, to_snake


def test_to_camel():
    assert to_camel("snake_case_words") == "snakeCaseWords"


def test_to_snake():
    assert to_snake("snakeCaseWords") == "snake_case_words"
