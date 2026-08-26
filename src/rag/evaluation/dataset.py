import json


class EvaluationDataset:


    def __init__(
        self,
        path
    ):

        with open(
            path,
            encoding="utf-8"
        ) as f:

            self.samples = json.load(f)



    def __iter__(self):

        return iter(
            self.samples
        )



    def __len__(self):

        return len(
            self.samples
        )