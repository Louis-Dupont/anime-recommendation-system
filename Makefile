.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = anime-recommendation-system
PYTHON_INTERPRETER = python

include .env
export

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# Initialise all environement variables from .env file
init_env_variables:
	export $(grep -v '^#' .env | xargs)


## Install Python Dependencies
requirements: test_environment
	python -m pip install -U pip setuptools wheel
	python -m pip install -r requirements.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


######################
#	DIRECT RUNNERS	 #
######################
train:
	python src/training.py \
		--data_folder $(PWD)/data \
		--model_path $(PWD)/models/trained_model.pkl \
		--nb_hidden_features $(NB_FEATURES)

flask_run:
	python flask/app.py \
		--model_path $(PWD)/models/trained_model.pkl \
		--debug False \
		--host_ip $(FLASK_IP) \
		--port $(FLASK_PORT)

streamlit_run:
	streamlit run streamlit/app.py \
		--browser.serverAddress $(STREAMLIT_IP) \
		--server.port $(STREAMLIT_PORT) \
		-- \
			--model_ip $(FLASK_IP) \
			--model_port $(FLASK_PORT)


######################
#	DOCKER BUILDS	 #
######################
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


######################
#	DOCKER RUN	 	#
######################
docker_run_training:
	docker run \
		-it --rm \
		-v $(PWD)/data/anime.csv:/anime/data/anime.csv \
		-v $(PWD)/data/rating_complete.csv:/anime/data/rating_complete.csv \
		training:v0 \
			--data_folder /anime/data \
			--model_path /models/trained_model_todelete_new.pkl \
			--nb_hidden_features 2

docker_run_flask:
	docker run \
		-p $(FLASK_PORT):$(FLASK_PORT) \
		-v $(PWD)/models:/anime/models \
		flask:v0 \
			--model_path /anime/models/trained_model.pkl

docker_run_streamlit:
	docker run \
		-it --rm \
		streamlit:v0


######################
#  EXAMPLES TO TEST	 #
######################
predict_shingeki_no_kyojin:
	@curl -X POST http://$(FLASK_IP):$(FLASK_PORT)/predict \
		-d @flask/samples/shingeki_no_kyojin.json \
		-H "Content-Type: application/json"

predict_mainstream:
	@curl -X POST http://$(FLASK_IP):$(FLASK_PORT)/predict \
		-d @flask/samples/mainstream.json \
		-H "Content-Type: application/json"
