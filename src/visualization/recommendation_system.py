import pathlib
from typing import List

import pandas as pd
import numpy as np
import pickle

from dataset import Dataset
from model import RatingModel, AnimeStandardScaler


class RecommendationSystem:

    def __init__(
        self,
        dataset,
        standardizer,
        model: RatingModel
        ):
        self.dataset = dataset
        self.standardizer = standardizer
        self.model = model

        anime_titles, anime_ids = dataset.df_titles.Name.apply(str.lower), dataset.df_titles.MAL_ID
        self.anime_title_to_id = dict(zip(anime_titles, anime_ids))
        self.anime_id_to_title = dict(zip(anime_ids, anime_titles))

        # self.exclude_rated_animes = exclude_rated_animes
        # self.columns_to_keep = columns_to_keep

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            return pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def predict_relevant_anime(self, anime_title_to_rating, exclude_rated_animes=True, columns_to_keep=['title', 'relevancy']):
        predicted_ratings = self._predict_single_user_ratings(anime_title_to_rating)
        df_predicted_ratings = self._build_user_estimated_df(predicted_ratings)
        df_predicted_ratings = self._compute_relevancy(df_predicted_ratings)
        df_predicted_ratings = self._clean_output(
            df_predicted_ratings=df_predicted_ratings,
            rated_animes=[_clean_anime_title(title) for title in anime_title_to_rating.keys()],
            exclude_rated_animes=self.exclude_rated_animes,
            features_to_keep=self.columns_to_keep
        )
        return df_predicted_ratings

    def _predict_single_user_ratings(self, anime_title_to_rating):
        user_ratings = self._init_user_rating_array(anime_title_to_rating)
        predicted_ratings = self.model.predict(
            data=user_ratings,
            standardize_input=True,
            destandardize_output=True
        )
        return predicted_ratings[0]

    def _init_user_rating_array(self, anime_title_to_rating):
        user_ratings = self.standardizer.mean_array[np.newaxis, :].copy()
        for anime_title, rating in anime_title_to_rating.items():
            anime_id = self.anime_title_to_id[_clean_anime_title(anime_title)]
            user_ratings[0, anime_id] = rating
        return user_ratings

    def _build_user_estimated_df(self, predicted_ratings):
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


    def _compute_relevancy(self, df_predicted_ratings):
        """Compute the relevancy of a given prediction based on 3 different criteria:
        1.popularity_score: measures in log scale to what extend each anime is rated.
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

        df_predicted_ratings['relevancy'] = 3 / ( \
            1/df_predicted_ratings.popularity_score + \
            1/df_predicted_ratings.personal_rating_score + \
            1/df_predicted_ratings.relative_rating_score \
        )
        return df_predicted_ratings

    def _clean_output(self, df_predicted_ratings, rated_animes, exclude_rated_animes, features_to_keep):
        if exclude_rated_animes:
            df_anime_to_drop = df_predicted_ratings['title'].apply(lambda x: x not in rated_animes)
            df_predicted_ratings = df_predicted_ratings[df_anime_to_drop]
        df_predicted_ratings = df_predicted_ratings[features_to_keep] 
        df_predicted_ratings = df_predicted_ratings.sort_values(by='relevancy', ascending=False)
        return df_predicted_ratings


def _clean_anime_title(title):
    """Preprocesing to not be case sensitive."""
    return title.lower()



if __name__ == '__main__':
    
    HOME_DIR = pathlib.Path().home()
    DATA_DIR = HOME_DIR / 'data/other/archive'

    df_titles = pd.read_csv(DATA_DIR / 'anime.csv')
    df_ratings = pd.read_csv(DATA_DIR / 'rating_complete.csv')

    save_to = 'test_save.pkl'


    nb_anime = df_ratings.anime_id.max()+1

    standardizer = AnimeStandardScaler(nb_anime=nb_anime)
    dataset = Dataset(df_titles, df_ratings, standardizer=standardizer, nb_anime=nb_anime, test_size=0.2)

    training_data = dataset.get_standardized_sample(test=False)
    model = RatingModel(
        standardizer=standardizer,
        n_components=100,
        n_iter=50,
        random_state=42
    ).fit(training_data)

    recommendation_system = RecommendationSystem(dataset, standardizer, model, exclude_rated_animes=True)
    recommendation_system.save(save_to)

    # HOME_DIR = pathlib.Path().home()
    # DATA_DIR = HOME_DIR / 'data/other/archive'

    # df_titles = pd.read_csv(DATA_DIR / 'anime.csv')
    # df_ratings = pd.read_csv(DATA_DIR / 'rating_complete.csv')

    # nb_anime = df_ratings.anime_id.max()+1

    # standardizer = AnimeStandardScaler(nb_anime=nb_anime)
    # dataset = Dataset(df_titles, df_ratings, standardizer=standardizer, nb_anime=nb_anime, test_size=0.2)

    # training_data = dataset.get_standardized_sample(test=False)
    # model = RatingModel(
    #     standardizer=standardizer,
    #     n_components=100,
    #     n_iter=50,
    #     random_state=42
    # ).fit(training_data)

    # estimator = RecommendationSystem(dataset, standardizer, model, exclude_rated_animes=True)

    # with open('estimator.pkl', 'wb') as f:
    #     pickle.dump(estimator, f)