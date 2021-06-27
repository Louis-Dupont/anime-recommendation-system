import requests
import json
import argparse
from typing import Dict, List

import streamlit as st
import pandas as pd
import altair as alt

import SessionState
from src.recommendation_system import (
    RecommendationSystem,
    get_unique_anime_names
)

HEADERS = {'content-type': 'application/json'}

GOOGLE_SEARCH_TEMPLATE = 'https://www.google.com/search?q={formated_anime}'
RECOMMENDATION_LINE_TEMPLATE = ' > {rank} - {anime} ([google]({anime_link}))'


@st.cache()
def fetch_prediction_results(
    recommendation_server_url: str,
    anime_ratings: Dict[str, int],
    exclude_rated_animes: bool = True,
    columns_to_keep: List[str] = RecommendationSystem.VARIOUS_FEATURES,
    scoring_weights: Dict[str, float] = RecommendationSystem.DEFAULT_SCORING_WEIGHTS,
    number_of_results: int = RecommendationSystem.DEFAULT_NB_RESULTS
    ) -> pd.DataFrame:
    data = json.dumps(dict(
        anime_ratings_json = anime_ratings.get_dict(),
        exclude_rated_animes = exclude_rated_animes,
        columns_to_keep = columns_to_keep,
        scoring_weights = scoring_weights,
        number_of_results = number_of_results
    ))

    recommendation_json = requests.post(
        recommendation_server_url + '/predict',
        data=data,
        headers=HEADERS
    ).json()

    recommendation_df = pd.DataFrame.from_records(recommendation_json)
    return recommendation_df


@st.cache()
def get_unique_anime_names_cached(anime_path):
    return get_unique_anime_names(anime_path)


class AnimeRatings:
    '''This class wraps an instance of SessionState
    which allows streamlit to keep in memory the anime
    ratings even after the page reloads, as opposed to
    other variables.'''

    def __init__(self):
        self.session_state = SessionState.get(anime_to_rating={})

    def get_dict(self) -> Dict[str, int]:
        return self.session_state.anime_to_rating

    def add_element(self, anime: str, rating: int) -> None:
        self.session_state.anime_to_rating[anime] = rating

    def reset(self) -> None:
        self.session_state.anime_to_rating = {}

    def is_not_empty(self) -> bool:
        return len(self.session_state.anime_to_rating) > 0


def init_layout() -> None:
    st.markdown('# Anime Recommendation Application')
    st.markdown('''
        ## How to use
        You can rate (0-10) animes you've watched on the
        left pane.\n
        The more animes you rate the more precise the recommendation will be.\n
        You can also play with 3 parameters used for the prediction to
        help the system emphasize more on a factor you find more important.
    ''')


def get_user_anime_ratings(
    anime_ratings: AnimeRatings,
    anime_names: List[str]
    ) -> None:
    with st.sidebar:
        st.markdown('# Personal Ratings')
        selected_anime = st.selectbox(
            label='What anime did you watch?',
            options=[''] + anime_names
        )
        rating = st.number_input(
            label='What rating would you give it? (0-10)',
            min_value=0,
            max_value=10,
            value=10
        )

        if selected_anime != '':
            add_rating = st.button(f'Add "{selected_anime}"')
            if add_rating:
                anime_ratings.add_element(anime=selected_anime, rating=rating)
        else:
            st.markdown('Please select an anime')

    if len(anime_ratings.get_dict()) ==  0:
        st.markdown('**Please rate at least one anime**')
        return

    with st.sidebar:
        reset_ratings = st.button('RESET ALL RATINGS')

    if reset_ratings:
        anime_ratings.reset()
        st.markdown('## All the past ratings have been cleared')
        return

    display_anime_ratings(anime_ratings)


def get_nb_recommendations() -> int:
    with st.sidebar:
        st.markdown('---')
        st.markdown('# Other Parameters')
        nb_recommendations = st.number_input(
            label='How many recommendations do you want?',
            min_value=1,
            max_value=None,
            value=10
        )
    return nb_recommendations


def get_scoring_weights() -> Dict[str, float]:
    weights = {}
    with st.sidebar:
        st.markdown('''
            ---
            # Recommendation Parameters
            You can customize the weight of factors
            used to estimate what you may like.
        ''')

        st.markdown('### Global Popularity')
        weights['popularity_score'] = st.number_input(
            label='This factor will favorise animes that many people watched.',
            min_value=0,
            max_value=None,
            value=1,
            key=0
        )

        st.markdown('### Personal Affinity')
        weights['personal_rating_score'] = st.number_input(
            label='This factor will favorise animes that you may like in general.',
            min_value=0,
            max_value=None,
            value=1,
            key=1
        )

        st.markdown('### Relative Affinity')
        weights['relative_rating_score'] = st.number_input(
            label='This factor will favorise animes that you may like more than other people.',
            min_value=0,
            max_value=None,
            value=1,
            key=2
        )

    if 0 \
        == weights['personal_rating_score'] \
        == weights['relative_rating_score'] \
        == weights['popularity_score']:
        return RecommendationSystem.DEFAULT_SCORING_WEIGHTS
    return weights


def display_anime_ratings(anime_ratings: AnimeRatings) -> None:
    st.markdown('## Rated animes')
    for anime, rating in anime_ratings.get_dict().items():
        _, col = st.beta_columns((1, 10))
        with col:
            st.markdown(f'\n > **{anime}** with a rating of **{rating}**')


def display_user_recommendations(
    recommendation_df: pd.DataFrame,
    nb_recommendations: int
    ) -> None:

    st.markdown('## Recommendations')
    selected_recommendations_df = recommendation_df.head(nb_recommendations)

    for i, anime in enumerate(selected_recommendations_df.title):
        anime_link = GOOGLE_SEARCH_TEMPLATE.format(
            formated_anime='+'.join(anime.split(' '))
        )
        anime_result = RECOMMENDATION_LINE_TEMPLATE.format(
            rank=i+1,
            anime=anime,
            anime_link=anime_link
        )

        _, col = st.beta_columns((1, 10))
        with col:
            st.markdown(anime_result)

    display_df = st.button('Display raw data')
    if display_df:
        st.dataframe(recommendation_df)


def main(recommendation_server_url: str, anime_path: str) -> None:
    anime_names = get_unique_anime_names_cached(anime_path)
    anime_ratings = AnimeRatings()

    init_layout()
    get_user_anime_ratings(anime_ratings, anime_names)
    scoring_weights = get_scoring_weights()
    nb_recommendations = get_nb_recommendations()

    if anime_ratings.is_not_empty():
        recommendation_df = fetch_prediction_results(
            recommendation_server_url=recommendation_server_url,
            anime_ratings=anime_ratings,
            scoring_weights=scoring_weights
        )
        display_user_recommendations(recommendation_df, nb_recommendations)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--flask_url',
        help='Flask app url',
        default='http://127.0.0.2:5000',
        type=str
    )
    parser.add_argument(
        '--anime_path',
        help='Path to "anime.csv"',
        default='data/anime.csv',
        type=str
    )
    known_args, unknown_args = parser.parse_known_args()

    main(
        recommendation_server_url=known_args.flask_url,
        anime_path=known_args.anime_path
    )
