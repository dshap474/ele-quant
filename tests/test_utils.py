from ele_quant import slugify


def test_slugify_basic():
    assert slugify("Hello World") == "hello-world"


def test_slugify_strips_non_alphanum():
    assert slugify("Python @ 3.10!") == "python-3-10"
