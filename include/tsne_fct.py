"""tsne_fct.py
Additional dimensionality reduction methods"""

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from ingest import DataFrame
from train import IngredientsRBM
from ingred2vec import Binarizer

def attempt(df: DataFrame, rbm: IngredientsRBM, mlb: Binarizer):
    """kmeans on space of embeddings, then single application of tsne (2d)"""
    flattened_ingreds = []

    for u in df.dropna().sample(200).sampled_words.iloc[:200].to_list():
        flattened_ingreds += u
    kmeans = KMeans(n_clusters=128, max_iter=24000)
    rbm_embeds = rbm.str_sample_hidden([[u] for u in flattened_ingreds], mlb)
    yostacity = kmeans.fit_predict(rbm_embeds)
    tsne = TSNE(n_iter=2500)
    embeds = tsne.fit_transform(rbm.str_sample_hidden(
        [[u] for u in flattened_ingreds], mlb))
    ingred_labels = [u for u in flattened_ingreds]
    one_label_df = pd.DataFrame({
        'x': embeds[:,0],
        'y': embeds[:,1],
        'c': yostacity,
        'label': ingred_labels}).groupby('label').agg(lambda x: list(x)[0])

    one_label_df = one_label_df.sample(200)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(one_label_df.x, one_label_df.y, c=one_label_df.c)
    for i, txt in enumerate(one_label_df.index):
        ax.annotate(txt, (one_label_df.x.iloc[i], one_label_df.y.iloc[i]))

def attempt_reduce(df: DataFrame, rbm: IngredientsRBM, mlb: Binarizer):
    """kmeans on tsne-reduced space of embeddings (128-d), then second application of tsne (2d)"""
    flattened_ingreds = []

    for u in df.dropna().sample(200).sampled_words.iloc[:200].to_list():
        flattened_ingreds += u

    kmeans = KMeans(n_clusters=12, max_iter=24000)

    rbm_embeds = rbm.str_sample_hidden([[u] for u in flattened_ingreds], mlb)

    tsne_1 = TSNE(n_iter=2500,n_components=128, method="exact")
    rbm_embeds_1 = tsne_1.fit_transform(rbm_embeds)

    # rbm_embeds
    yostacity = kmeans.fit_predict(rbm_embeds_1)
    tsne = TSNE(n_iter=2500)
    embeds = tsne.fit_transform(rbm.str_sample_hidden(
        [[u] for u in flattened_ingreds], mlb))
    ingred_labels = [u for u in flattened_ingreds]
    one_label_df = pd.DataFrame({
        'x': embeds[:,0],
        'y': embeds[:,1],
        'c': yostacity,
        'label': ingred_labels}).groupby('label').agg(lambda x: list(x)[0])

    one_label_df = one_label_df.sample(200)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(one_label_df.x, one_label_df.y, c=one_label_df.c)
    for i, txt in enumerate(one_label_df.index):
        ax.annotate(txt, (one_label_df.x.iloc[i], one_label_df.y.iloc[i]))

def attempt_reduce_sequential_barnes(df: DataFrame, rbm: IngredientsRBM, mlb: Binarizer):
    """kmeans on tsne-reduced space of embeddings (3-d), then second application of tsne (2d)"""
    flattened_ingreds = []

    for u in df.dropna().sample(200).sampled_words.iloc[:200].to_list():
        flattened_ingreds += u
    kmeans = KMeans(n_clusters=12, max_iter=24000)

    rbm_embeds = rbm.str_sample_hidden([[u] for u in flattened_ingreds], mlb)

    tsne_1 = TSNE(n_iter=2500,n_components=3)
    rbm_embeds_1 = tsne_1.fit_transform(rbm_embeds)

    yostacity = kmeans.fit_predict(rbm_embeds_1)
    tsne = TSNE(n_iter=2500)
    embeds = tsne.fit_transform(rbm_embeds_1)
    ingred_labels = [u for u in flattened_ingreds]
    one_label_df = pd.DataFrame({
        'x': embeds[:,0],
        'y': embeds[:,1],
        'c': yostacity,
        'label': ingred_labels}).groupby('label').agg(lambda x: list(x)[0])

    one_label_df = one_label_df.sample(200)
    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(one_label_df.x, one_label_df.y, c=one_label_df.c)
    for i, txt in enumerate(one_label_df.index):
        ax.annotate(txt, (one_label_df.x.iloc[i], one_label_df.y.iloc[i]))
