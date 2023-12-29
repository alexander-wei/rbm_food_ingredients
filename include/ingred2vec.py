"""ingred2vec.py"""

from typing import List, Iterable, Tuple
from types import NoneType
from sklearn.preprocessing import MultiLabelBinarizer
from ingest import DataFrame
from util import split_iter
import numpy as np
from joblib import numpy_pickle

class Binarizer(MultiLabelBinarizer):
    """generate multi-hot vector for each recipe, with 1's indicating ingredient presence"""
    dataframe = None

    def __init__(self, dataframe: DataFrame, classes: List | None = None, sparse_output: bool = False) -> None:
        super().__init__(classes=classes, sparse_output=sparse_output)
        self.fit(dataframe.sampled_words)

    @staticmethod
    def load(dataframe: DataFrame, mlb: MultiLabelBinarizer | NoneType = None):
        if isinstance(mlb, NoneType):
            mlb = Binarizer(dataframe=dataframe)
        xtrain = mlb.transform(dataframe.sampled_words)

        return xtrain.astype(np.float16)

class BatchedDataFrame:
    """convenience methods for generating dataset that can be streamed by torch DataLoader"""
    label_binarizer: Binarizer
    iterator: Iterable

    @staticmethod
    def batches(data: DataFrame, batch_size=36, *ac, **av) -> Tuple[Binarizer, Iterable]:
        label_binarizer = Binarizer(data.dropna())
        iterator = split_iter(Binarizer.load(data.dropna(), label_binarizer), batch_size)

        return label_binarizer, iterator
    
    @staticmethod
    def save(df: DataFrame, batch_size=36, dest="."):
        _, batches = BatchedDataFrame.batches(df, batch_size)
        numpy_pickle.dump(list(batches), "./out/batches_%d_%d.job" % (0,batch_size))
