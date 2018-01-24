import random
import tempfile

import jsonlines
import requests

IRIS_DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

IRIS_COLUMNS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species"]
IRIS_SPECIES = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}


def load_iris_from_uci():
    all_iris_lines = [l for l in requests.get(IRIS_DATA_URL).text.splitlines() if len(l) > 0]
    assert len(all_iris_lines) == 150

    return all_iris_lines


def get_iris_as_jsons():

    def _iris_line_to_dict(l):
        items = l.split(',')
        map = dict()
        for idx, i in enumerate(items):
            if idx != len(items) - 1:
                map[IRIS_COLUMNS[idx]] = i
            else:
                map[IRIS_COLUMNS[idx]] = IRIS_SPECIES[i]

        return map

    all_iris_lines = load_iris_from_uci()
    all_iris_jsons = [jl for jl in map(_iris_line_to_dict, all_iris_lines)]

    return all_iris_jsons


def get_iris_as_jsonlines_file():
    all_iris_jsons = get_iris_as_jsons()

    _, iris_jsonl_path = tempfile.mkstemp("iris.json", text=True)
    with jsonlines.open(iris_jsonl_path, mode="w") as f:
        for j in all_iris_jsons:
            f.write(j)

    return iris_jsonl_path


def create_iris_train_test_datafiles(split=0.8, shuffle=True):
    all_iris_jsons = get_iris_as_jsons()

    split_at = int(len(all_iris_jsons) * split)
    if shuffle:
        random.shuffle(all_iris_jsons)

    train_iris_jsons = all_iris_jsons[:split_at]
    test_iris_jsons = all_iris_jsons[split_at:]

    filepaths = {}
    for segment, jsons in [("train", train_iris_jsons), ("test", test_iris_jsons)]:
        _, iris_jsonl_path = tempfile.mkstemp("iris_%s.json" % segment, text=True)
        with jsonlines.open(iris_jsonl_path, mode="w") as f:
            for j in jsons:
                f.write(j)

        filepaths[segment] = iris_jsonl_path

        print("iris %s jsonlines written to %s as follows:\n%s" % (
            segment, filepaths[segment], open(filepaths[segment], mode="r").read(-1)))

    train_iris_jl_path = filepaths["train"]
    test_iris_jl_path = filepaths["test"]

    return train_iris_jl_path, test_iris_jl_path

