.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = anime-recommendation-system
PYTHON_INTERPRETER = python

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


train:
	python src/training.py \
		--data_folder $(PWD)/data \
		--model_path $(PWD)/models/trained_model_todelete.pkl \
		--nb_hidden_features 2


docker_build_training:
	docker build \
		-t training:v0 \
		-f docker/training/Dockerfile .

docker_build_flask:
	docker build \
		-t flask:v0 \
		-f docker/flask/Dockerfile .

docker_build_streamlit:
	docker build \
		-t streamlit:v0 \
		-f docker/streamlit/Dockerfile .


PROJECT_FOLDER=/Users/louisdupont/Desktop/python_tests/anime-recommendation-system
docker_run_training:
	docker run \
		-it --rm \
		-v $(PROJECT_FOLDER)/data/anime.csv:/anime/data/anime.csv \
		-v $(PROJECT_FOLDER)/data/rating_complete.csv:/anime/data/rating_complete.csv \
		training:v0 \
			--data_folder /anime/data \
			--model_path /models/trained_model_todelete_new.pkl \
			--nb_hidden_features 2


docker_run_flask:
	docker run \
		-p 5000:5000 \
		-v $(PROJECT_FOLDER)/models:/anime/models \
		flask:v0 \
			--model_path /anime/models/trained_model.pkl


docker_run_streamlit:
	docker run \
		-it --rm \
		streamlit:v0


flask_run:
	python flask/app.py\
		--model_path /Users/louisdupont/Desktop/python_tests/anime-recommendation-system/models/trained_model.pkl

streamlit_run:
	streamlit run  streamlit/app.py -- \
		--anime_path data/anime.csv \
		--flask_url http://172.17.0.2:5000


predict_shingeki_no_kyojin:
	@curl -X POST http://172.17.0.2:5000/predict \
		-d @flask/samples/shingeki_no_kyojin.json \
		-H "Content-Type: application/json"


predict_mainstream:
	@curl -X POST http://127.0.0.1:5000/predict \
		-d @flask/samples/mainstream.json \
		-H "Content-Type: application/json"
