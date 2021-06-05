import argparse
import pandas as pd
from src.recommendation_system import RecommendationSystem


def train(data_dir, save_to, nb_hidden_features):
    print('STARTING')
    df_titles = pd.read_csv(f'{data_dir}/anime.csv')
    df_ratings = pd.read_csv(f'{data_dir}/rating_complete.csv')
    print('LOADED')
    recommendation_system = RecommendationSystem(
        df_titles=df_titles,
        df_ratings=df_ratings,
        n_components=nb_hidden_features
    )
    print('READY')
    recommendation_system.train(
        n_iter=2,
        random_state=42
    )
    print('TRAINED')
    recommendation_system.save(save_to)
    print('SAVED')


if __name__ == '__main__':

    
    from os import listdir
    from os.path import isfile, join
    
    print([f for f in listdir() if isfile(f)])
    print([f for f in listdir('src') if isfile(join('src', f))])
    print([f for f in listdir('data') if isfile(join('data', f))])

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_folder',
        help='Where the data is stored',
        type=str
    )
    parser.add_argument(
        '--model_path',
        help='How you want to name the saved model',
        type=str
    )
    parser.add_argument(
        '--nb_hidden_features',
        help='The size of the encoder',
        type=int
    )
    known_args, unknown_args = parser.parse_known_args()
    print(vars(known_args))
    train(
        data_dir=known_args.data_folder,
        save_to=known_args.model_path,
        nb_hidden_features=known_args.nb_hidden_features
    )
    print('DONE')
