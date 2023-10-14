from data_pipeline import pipeline


def test_pipeline():
    pipeline("data/data.txt")
    assert True
