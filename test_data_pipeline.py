from data_pipeline import pipeline


def test_pipeline():
    pipeline("data/token_1.txt")
    assert True
