from hello_uv import hello


def test_hello()->None:
    assert hello() == "Hello, world!"
