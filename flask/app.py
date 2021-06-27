import argparse
from flask import Flask, request, json, jsonify
from time import time
from src.recommendation_system import load, RecommendationSystem


app = Flask(__name__)


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        df = recommendation_system.predict_relevant_anime(
            anime_title_to_rating=data['anime_ratings_json'],
            exclude_rated_animes=data['exclude_rated_animes'],
            columns_to_keep=data['columns_to_keep'],
            scoring_weights=data['scoring_weights'],
            nb_results=data['number_of_results']
        )
        return df.to_json(orient='records')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        help='How you want to name the saved model',
        type=str
    )
    known_args, unknown_args = parser.parse_known_args()
    print(known_args.model_path)
    recommendation_system = RecommendationSystem.load(known_args.model_path)

    app.run(debug=True, host='0.0.0.0')
