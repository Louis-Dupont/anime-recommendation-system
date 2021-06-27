import pathlib
from typing import Dict, List
import argparse

import pandas as pd
import numpy as np
import pickle

from src.model import RecommendationModel, AnimeStandardScaler
from src.dataset import Dataset


class RecommendationSystem:
    '''Wraps everything used to recommand animes.
    It can be saved in pickle format to be used later.
    It does not require the initial dataset anymore once trained.
    '''

    DEFAULT_NB_RESULTS = 100
    MAIN_FEATURES = ['title', 'relevancy']
    OTHER_SCORING_FEATURES = [
        'popularity_score',
        'personal_rating_score',
        'relative_rating_score',
    ]
    VARIOUS_FEATURES = MAIN_FEATURES + OTHER_SCORING_FEATURES
    DEFAULT_SCORING_WEIGHTS = dict(
        popularity_score = 1,
        personal_rating_score = 1,
        relative_rating_score = 1
    )

    def __init__(self, df_titles: pd.DataFrame, df_ratings: pd.DataFrame, n_components: int) -> None:
        nb_anime = df_ratings.anime_id.max() + 1
        standardizer = AnimeStandardScaler(nb_anime=nb_anime)
        dataset = Dataset(df_titles, df_ratings, standardizer=standardizer, nb_anime=nb_anime, test_size=0.2)

        anime_titles, anime_ids = dataset.df_titles.Name.apply(str.lower), dataset.df_titles.MAL_ID

        self.anime_title_to_id = dict(zip(anime_titles, anime_ids))
        self.anime_id_to_title = dict(zip(anime_ids, anime_titles))
        self.n_components = n_components
        self.dataset = dataset
        self.standardizer = standardizer

    def train(self, n_iter: int = 50, random_state: int = 42):
        training_data = self.dataset.get_standardized_sample(test=False)
        self.model = RecommendationModel(
            standardizer=self.standardizer,
            n_components=self.n_components,
            n_iter=n_iter,
            random_state=random_state
        ).fit(training_data)

    def save(self, file_path: str):
        with open(file_path, 'wb') as f:
            return pickle.dump(self, f)

    @staticmethod
    def load(file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def predict_relevant_anime(
        self,
        anime_title_to_rating: Dict[str, int],
        exclude_rated_animes: bool,
        columns_to_keep: List[str],
        scoring_weights: Dict[str, float],
        nb_results: int
        ) -> pd.DataFrame:
        predicted_ratings = self._predict_single_user_ratings(anime_title_to_rating)
        df_predicted_ratings = self._build_user_estimated_df(predicted_ratings)
        df_predicted_ratings = self._compute_relevancy(df_predicted_ratings, scoring_weights)
        df_predicted_ratings = self._clean_output(
            df_predicted_ratings=df_predicted_ratings,
            rated_animes=[_clean_anime_title(title) for title in anime_title_to_rating.keys()],
            exclude_rated_animes=exclude_rated_animes,
            features_to_keep=columns_to_keep,
            nb_results=nb_results
        )
        return df_predicted_ratings

    def _predict_single_user_ratings(self, anime_title_to_rating: Dict[str, int]) -> np.ndarray:
        user_ratings = self._init_user_rating_array(anime_title_to_rating)
        predicted_ratings = self.model.predict(
            data=user_ratings,
            standardize_input=True,
            destandardize_output=True
        )
        return predicted_ratings[0]

    def _init_user_rating_array(self, anime_title_to_rating: Dict[str, int]) -> np.ndarray:
        user_ratings = self.standardizer.mean_array[np.newaxis, :].copy()
        for anime_title, rating in anime_title_to_rating.items():
            anime_id = self.anime_title_to_id[_clean_anime_title(anime_title)]
            user_ratings[0, anime_id] = rating
        return user_ratings

    def _build_user_estimated_df(self, predicted_ratings: np.ndarray) -> pd.DataFrame:
        anime_title, anime_id = zip(*self.anime_title_to_id.items())

        df_predicted_ratings = pd.DataFrame({
            'anime_id': range(len(predicted_ratings)),
            'user_estimated_rating': predicted_ratings,
            'users_average_rating': self.standardizer.mean_array
        })
        df_titles = pd.DataFrame({
            'anime_id': anime_id,
            'title': anime_title
        })
        df_predicted_ratings = pd.merge(df_predicted_ratings, df_titles, how="inner", on='anime_id')

        return df_predicted_ratings


    def _compute_relevancy(self, df_predicted_ratings: pd.DataFrame, scoring_weight: Dict[str, int]) -> pd.DataFrame:
        """Compute the relevancy of a given prediction based on 3 different criteria:
        1.popularity_score: measures in log scale how often each anime is rated.
        2. personal_rating_score: estimated score of each anime according to the user inputs.
        3. relative_rating_score: difference in the estimated score of the user and of other users.
        Each score is a float value in [0, 1].
        The harmonic mean of these scores is the relevancy.
        """
        df_predicted_ratings = pd.merge(self.standardizer.df_count, df_predicted_ratings, how="inner", on='anime_id')

        rating_count_log = df_predicted_ratings.rating_count.apply(np.log)
        df_predicted_ratings['popularity_score'] = rating_count_log / rating_count_log.max()

        df_predicted_ratings['personal_rating_score'] = df_predicted_ratings.user_estimated_rating / 10

        rating_difference = df_predicted_ratings.user_estimated_rating - df_predicted_ratings.users_average_rating
        df_predicted_ratings['relative_rating_score'] = \
            (rating_difference - rating_difference.min())/(rating_difference.max() - rating_difference.min())

        sum_scoring_weights = \
            scoring_weight['popularity_score'] \
            + scoring_weight['personal_rating_score'] \
            + scoring_weight['relative_rating_score']
        sum_scoring = \
            scoring_weight['popularity_score']/df_predicted_ratings['popularity_score'] \
            + scoring_weight['personal_rating_score']/df_predicted_ratings['personal_rating_score'] \
            + scoring_weight['relative_rating_score']/df_predicted_ratings['relative_rating_score']

        df_predicted_ratings['relevancy'] = sum_scoring_weights / sum_scoring
        return df_predicted_ratings

    def _clean_output(
        self,
        df_predicted_ratings: pd.DataFrame,
        rated_animes: List[str],
        exclude_rated_animes: bool,
        features_to_keep: List[str],
        nb_results: int
        ):
        if exclude_rated_animes:
            df_anime_to_drop = df_predicted_ratings['title'].apply(lambda x: x not in rated_animes)
            df_predicted_ratings = df_predicted_ratings[df_anime_to_drop]
        df_predicted_ratings = df_predicted_ratings[features_to_keep] 
        df_predicted_ratings = df_predicted_ratings.sort_values(by='relevancy', ascending=False)
        return df_predicted_ratings.head(nb_results)



def _clean_anime_title(title: str) -> str:
    """Preprocesing to not be case sensitive."""
    return title.lower()


def load(model_path: str) -> RecommendationSystem:
    return RecommendationSystem.load(file_path=model_path)


def get_unique_anime_names(path: str) -> List[str]:
    anime_df = pd.read_csv(path)
    anime_names = anime_df.Name.unique().tolist()
    return anime_names
