from typing import List

import pandas as pd
import numpy as np

from scipy import sparse
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


class RecommendationModel:
    """SVD based model (though also includes a "mean" functionality as extra feature)

    The SVD prediction works by learning a factorization matrix.
        - This can be seen as a simple encoder decoder.
        - The data is normalized before and the missing input values are set to 0 afterward
        in order to avoid their influence.
        - The normalized data is encoded and then decoded, and its output is denormalized.
        - The denormalized output corresponds to an estimation of what the users may like 
        based on his input ratings.
    
    The data should be a np.ndarray of shape [nb_user x nb_anime]
    """
    def __init__(self, standardizer, n_components=10, n_iter=50, random_state=42):
        self.standardizer = standardizer
        self.svd = TruncatedSVD(n_components=n_components, n_iter=n_iter, random_state=random_state)

    def fit(self, data: np.ndarray, standardize_input=False):
        """Fit the SVS model"""
        if standardize_input:
            data = self.standardizer.transform_array(data)
        embedding_test = self.svd.fit(data)
        return self

    def predict(self, data: np.ndarray, standardize_input=False, destandardize_output=True):
        """SVD based prediction (each anime will its mean rating over all the users)."""
        if standardize_input:
            data = self.standardizer.transform_array(data)

        pred_data = self.svd.inverse_transform(self.svd.transform(data))

        if not destandardize_output:
            return pred_data

        pred = self.standardizer.reverse_transform_array(pred_data)
        pred[pred<0], pred[pred>10] = 0, 10
        return pred

    def mean_predict(self, data: np.ndarray, destandardize_output: bool = True):
        """Mean based prediction (each anime will its mean rating over all the users)."""
        if not destandardize_output:
            return np.zeros(data.shape)
        return self.standardizer.mean_array[np.newaxis, :].repeat(data.shape[0], 0)


class AnimeStandardScaler:
    """Hybrid class to standardize both dataframes and arrays.
    The means and std are computed on animes scoring.
    """

    def __init__(self, nb_anime):
        self.nb_anime = nb_anime

    def fit(self, df_train):
        """Store mean and std of the training dataframe in dataframe and array.
        """
        self._fit_df(df_train)
        self._fit_array()
        return self

    def _fit_df(self, df_train):
        """Store mean, std and count dataframes based on training data.
        """
        self.df_mean = df_train.groupby('anime_id', as_index=False).rating.mean().\
            rename(columns={'rating': 'rating_mean'})
        self.df_std = df_train.groupby('anime_id', as_index=False).rating.std(ddof=0).\
            rename(columns={'rating': 'rating_std'}).replace(0, 1)
        self.df_count = df_train.groupby('anime_id', as_index=False).rating.count().\
            rename(columns={'rating': 'rating_count'})

    def _fit_array(self):
        """Store mean and std arrays based on the previously computed dataframes.
        The index of these arrays correspond to the index of the anime,
        and the values correspond to the ratings mean/std of this anime.
        """
        mean_array = np.zeros(self.nb_anime)
        mean_array[self.df_mean.anime_id] = self.df_mean.rating_mean
        mean_array = np.where(mean_array==0, mean_array[mean_array != 0].mean(), mean_array) 

        std_array = np.zeros(self.nb_anime)
        std_array[self.df_std.anime_id] = self.df_std.rating_std
        std_array = np.where(std_array==0, std_array[std_array != 0].mean(), std_array)

        self.mean_array, self.std_array = mean_array, std_array

    def reverse_transform_array(self, array):
        """Unstandardize an array."""
        return array*self.std_array[np.newaxis, :] + self.mean_array[np.newaxis, :]

    def transform_array(self, array):
        """Standardize an array."""
        return (array - self.mean_array[np.newaxis, :])/self.std_array[np.newaxis, :]

    def transform_df(self, df):
        """Add 'standardized_rating' column to the input df, corresponding to the scoring
        after standardizing its values.
        """
        df = pd.merge(df, self.df_mean, how="inner", on='anime_id')
        df = pd.merge(df, self.df_std, how="inner", on='anime_id')
        df['standardized_rating'] = (df.rating - df.rating_mean)/df.rating_std
        return df
