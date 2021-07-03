#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = anime-recommendation-system
LOCAL_IP_ADRESS := $(shell ipconfig getifaddr en0)

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
	pip install -r requirements.txt


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
		--host_ip $(LOCAL_IP_ADRESS) \
		--port $(FLASK_PORT)

streamlit_run:
	streamlit run streamlit/app.py \
		--browser.serverAddress $(LOCAL_IP_ADRESS) \
		--server.port $(STREAMLIT_PORT) \
		-- \
			--anime_path $(PWD)/data/anime.csv \
			--model_ip $(LOCAL_IP_ADRESS) \
			--model_port $(FLASK_PORT)


######################
#	  DOCKER 	 	#
######################
docker_build_training:
	docker build \
		-t training:v0 \
		-f docker/training/Dockerfile .

docker_run_training:
	docker run \
		-it --rm \
		-v $(PWD)/data:/anime/data \
		-v $(PWD)/models:/anime/models \
		training:v0 \
			--data_folder /anime/data \
			--model_path /models/trained_model_todelete_new.pkl \
			--nb_hidden_features 2


docker_build_flask:
	docker build \
		-t flask:v0 \
		-f docker/flask/Dockerfile .

docker_run_flask:
	docker run \
		-p $(FLASK_PORT):$(FLASK_PORT) \
		-v $(PWD)/models:/anime/models \
		flask:v0 \
			--model_path=./models/trained_model.pkl \
			--debug False \
			--host_ip $(LOCAL_IP_ADRESS) \
			--port $(FLASK_PORT)


docker_build_streamlit:
	docker build \
		-t streamlit:v0 \
		-f docker/streamlit/Dockerfile .

docker_run_streamlit:
	docker run \
		-p $(STREAMLIT_PORT):$(STREAMLIT_PORT) \
		-v $(PWD)/data:/anime/data \
		streamlit:v0 \
			--browser.serverAddress $(LOCAL_IP_ADRESS) \
			--server.port $(STREAMLIT_PORT) \
			-- \
				--anime_path ./data/anime.csv \
				--model_ip $(LOCAL_IP_ADRESS) \
				--model_port $(FLASK_PORT)


######################
#  EXAMPLES TO TEST	 #
######################
predict_shingeki_no_kyojin:
	curl -X POST http://$(LOCAL_IP_ADRESS):$(FLASK_PORT)/predict \
		-d @flask/samples/shingeki_no_kyojin.json \
		-H "Content-Type: application/json"

predict_mainstream:
	@curl -X POST http://$(LOCAL_IP_ADRESS):$(FLASK_PORT)/predict \
		-d @flask/samples/mainstream.json \
		-H "Content-Type: application/json"
