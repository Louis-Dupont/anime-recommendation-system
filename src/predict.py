from src.recommendation_system import RecommendationSystem

def load(model_path):
    return RecommendationSystem.load(file_path=model_path)
