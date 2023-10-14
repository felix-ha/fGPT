from data_pipeline import pipeline


def test_pipeline():
    pipeline("data/data_train.txt", "data/data_validation.txt")
    assert True
