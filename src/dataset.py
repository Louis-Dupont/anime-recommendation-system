import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy import sparse
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

class Dataset:
    """Store the data as pd.DataFrame, as well as scipy sparse matrices.

    Notes:
    - The sparse matrices are standardized to increase the effiency of the model. 
    The data had to be standardized before the creation of the sparse matrix to
    keep its sparsity.
    - The train/test sampling is done in a deterministic way because if the data stucture.
    The data for each user is writtin thr
    """

    def __init__(self, df_titles, df_ratings, standardizer, nb_anime, test_size=0.1, random_state=42):
        self.nb_anime = nb_anime
        self.df_titles = df_titles
        self.df_train, self.df_test = self._train_test_split(
            df_ratings=df_ratings,
            test_size=test_size,
            random_state=random_state
        )

        standardizer = standardizer.fit(self.df_train)
        self.df_train = standardizer.transform_df(self.df_train)
        self.df_test = standardizer.transform_df(self.df_test)
        self.csr_train_standardized = self._df_to_csr(self.df_train)
        self.csr_test_standardized = self._df_to_csr(self.df_test)


    def get_standardized_sample(self, nb_sample=-1, test=False):
        """Get some samples, not randomized."""
        if test:
            samples = self.csr_test_standardized[:nb_sample]            
        else:
            samples = self.csr_train_standardized[:nb_sample]
        return samples

    def _df_to_csr(self, df):
        """Transform a dataframe of ratings into 'csr' sparse matrix."""
        data = df.standardized_rating
        rows, cols = preprocessing.LabelEncoder().fit_transform(df.user_id), df.anime_id
        shape = len(df.user_id.unique()), self.nb_anime
        return sparse.csr_matrix((data, (rows, cols)), shape)

    def _train_test_split(self, df_ratings, test_size, random_state):
        train_inds, test_inds = next(
            GroupShuffleSplit(
                test_size=test_size,
                n_splits=2,
                random_state=random_state
            ).split(df_ratings, groups=df_ratings['user_id'])
        )
        return df_ratings.iloc[train_inds], df_ratings.iloc[test_inds]


def score(y_pred, y_true_standardized, standardizer):
    mask = (y_true_standardized == 0)

    y_true = standardizer.reverse_transform_array(y_true_standardized)[~mask]
    y_pred = y_pred[~mask]

    y_delta = y_pred-y_true
    score_dict = {
        'accuracy':(np.round(y_pred)==np.round(y_true)).mean(),
        'mean_distance': (np.abs(y_delta)).mean(),
        'mean_squared_distance': np.sqrt(np.square(y_delta).mean())
    }

    return score_dict

