import argparse
from flask import Flask, request, json, jsonify
import fire
from src.recommendation_system import load, RecommendationSystem


app = Flask(__name__)

def build_prediction_route(recommendation_system: RecommendationSystem):

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


def run(model_path: str, debug: bool, host_ip: str, port: int):
    """Run a flask API.

    Args:
        model_path (str): Where you want to name the saved model
        debug (bool): Whether or not you want to enable debugging mode
        host_ip (str): Host IP adress
        port (int): Host port
    """

    recommendation_system = RecommendationSystem.load(model_path)
    print('loaded')
    build_prediction_route(recommendation_system)
    print('function_loaded')
    app.run(debug=debug, host=host_ip, port=port)

if __name__ == '__main__':
    fire.Fire(run)
