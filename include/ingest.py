"""ingest.py
data intake and preprocessing"""

from ast import literal_eval
from random import sample, randint

import joblib
from tqdm.auto import tqdm
from pandas import DataFrame as pdf, read_csv, Series
from torch.utils.data import Dataset as TorchDataset
from torch import from_numpy

from util import collect_first_k_words

def try_literal_eval(s):
    try:
        return literal_eval(s)
    except: return []

class DataFrame(pdf):
    """DataFrame structure for loading ingredient lists from the Food.com dataset
    Extends pandas.DataFrame
    Implements preprocess(), collect_first_words() methods for parsing raw ingredient data
    """
    ingredients: Series
    sampled_words: Series
    id: Series

    def __init__(self, data: str | pdf, *ac, **av) -> None:
        """init from path or pandas.dataframe"""
        if isinstance(data, str):
            data = DataFrame.read_csv(data, *ac, **av)

        super().__init__(data=data)
        self.preprocess()
        self.collect_first_words()

    @property
    def _constructor(self):
        return DataFrame

    @property
    def df(self):
        """recast as pandas dataframe for __repr__() compatibility"""
        return pdf(self)

    def dropna(self, *ac, **av):
        return pdf(self).dropna(*ac, **av)

    @staticmethod
    def read_csv(path, *ac, **av):
        df = read_csv(path, *ac, **av)
        return df

    def collect_first_words(self):
        """the first two words in every ingredient that is listed is selected as a
        distinct ingredient: ["cherry pie", "chardonnay"] -> [cherry, pie, chardonnay]"""
        ingr_aggregator = []

        self['sampled_words'] = Series()
        idx_ingredients = list(zip(self.id, self.ingredients))
        for idx, ingred_list\
            in tqdm(idx_ingredients):
            # recipe_id, list of ingredients per recipe
            for ingr_group in sample(ingred_list, randint(0,len(ingred_list))):
                # set of ingredients subset of all ingredients (sample)
                for w in collect_first_k_words(ingr_group, 2):
                    # first two words (split ' ') of each ingredient is entered
                    ingr_aggregator.append({'id': idx,
                        'ingredients': w})
        parsed_ingrs = pdf(ingr_aggregator)
        parsed_ingr_groups = parsed_ingrs.groupby("id").agg(list)
        idx_lookup = self.df.reset_index().set_index('id').to_dict()['index']

        for idx, ingreds in tqdm(parsed_ingr_groups.iterrows(),
                                 total=len(parsed_ingr_groups)):
            self.at[idx_lookup[idx], 'sampled_words'] = ingreds.to_list()[0]

    def preprocess(self):
        """apply literal_eval to convert raw unicode str to python objects"""
        ingredients = self.ingredients.apply(try_literal_eval)
        self['ingredients'] = ingredients

class Dataset(TorchDataset):
    """dataset from csv"""
    def __init__(self, feature_file, *ac):
        self.feature_file = feature_file

    def __len__(self):
        """This returns the total number of batches"""
        return len(joblib.load(self.feature_file))

    def __getitem__(self, idx):
        """Loading data on the fly (streaming)"""
        X = joblib.load(self.feature_file)[idx]

        return from_numpy(X).float()
