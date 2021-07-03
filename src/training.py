import pandas as pd
import fire

from src.recommendation_system import RecommendationSystem


def train(data_folder, model_path, nb_hidden_features):
    df_titles = pd.read_csv(f'{data_folder}/anime.csv')
    df_ratings = pd.read_csv(f'{data_folder}/rating_complete.csv')
    recommendation_system = RecommendationSystem(
        df_titles=df_titles,
        df_ratings=df_ratings,
        n_components=nb_hidden_features
    )
    recommendation_system.train(
        n_iter=2,
        random_state=42
    )
    recommendation_system.save(model_path)


if __name__ == '__main__':
    fire.Fire(train)
