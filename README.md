ANIME RECOMMENDATION APPLICATION
==============================

Project Organization
------------

    ├── docker             <- folder with all the useful Dockerfiles
    │
    ├── flask              <- Flask API used to expose the model
    │
    ├── models             <- Trained and serialized models
    │
    ├── src                <- Source code for use in this project.
    │
    ├── streamlit          <- Streamlit user interface 
    │
    ├── Makefile           <- Makefile with commands like `make train`
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# How to run the project
## 1. Environment Initialization
### 1.1. Python requirements
```console
pip install -r requirements.txt
```

### 1.2. Environment Variables
The variables used in this ``README`` are all defined in the ``.env`` file.<br/>
These variables are automatically imported in the Makefile but if in the following parts you want to run any of the command manually you have to run either of these command lines
```console
make init_env_variables
```
or with the full command
```console
export $(grep -v '^#' .env | xargs)
```
In both cases, you also need to store your local IP adress.
```console
LOCAL_IP_ADRESS=$(ipconfig getifaddr en0)
```
This needs to be run **every time** you start a new terminal.

## 2. Get the data
The data used in this project comes from a Kaggle dataset *[Anime Recommendation Database 2020](https://www.kaggle.com/hernan4444/anime-recommendation-database-2020)*. <br/>
For the training of this app you will need ``anime.csv`` and ``rating_complete.csv`` to be stored in the ``./data/`` folder.


## 3. Train the model
```console
make train
```
or with the full command
```console
python src/training.py \
    --data_folder $(PWD)/data \
    --model_path $(PWD)/models/trained_model.pkl \
    --nb_hidden_features $NB_FEATURES
```
This will take a couple of minutes depending on your own computer configuration but also on the number of features used.


## 4. Deploy the model
### 4.1 Launch the Flask API
```console
make flask_run
```
or with the full command
```console
python flask/app.py \
    --model_path=$(PWD)/models/trained_model.pkl \
    --debug=False \
    --host_ip=$LOCAL_IP_ADRESS \
    --port=$FLASK_PORT
```
The server should be ready to use in a few seconds.
### 4.2 Test it (optional)
```console
make predict_shingeki_no_kyojin
```
or with the full command
```console
curl -X POST http://$LOCAL_IP_ADRESS:$FLASK_PORT/predict \
    -d @flask/samples/shingeki_no_kyojin.json \
    -H "Content-Type: application/json"
```
With this you should get a recommendation based on **Shingeki no Kyojin** (Attack on Titans) ratings.

## 5. Deploy the user interface
### 5.1 Launch the streamlit app
```console
make streamlit_run
```
or with the full command
```console
streamlit run streamlit/app.py \
    --browser.serverAddress $LOCAL_IP_ADRESS \
    --server.port $STREAMLIT_PORT \
    -- \
        --anime_path $(PWD)/data/anime.csv \
        --model_ip $LOCAL_IP_ADRESS \
        --model_port $FLASK_PORT
```
### 5.2 Try it out!
If no tab started on your brower you can run this command and paste the result in your favorite browser.
```console
echo http://$STREAMLIT_IP:$STREAMLIT_PORT
```

# 6. Running previous services with Docker
## 6.0 Install Docker
[Install docker here](https://docs.docker.com/get-docker/)
## 6.1 Training
### 6.1.1 Build
```console
make docker_build_training
```
or with the full command
```console
docker build \
    -t training:v0 \
    -f docker/training/Dockerfile .
```
### 6.1.2 Run
```console
make docker_run_training
```
or with the full command
```console
docker run \
    -it --rm \
    -v $(PWD)/data:/anime/data \
    -v $(PWD)/models:/anime/models \
    training:v0 \
        --data_folder /anime/data \
        --model_path /models/trained_model.pkl \
        --nb_hidden_features $NB_FEATURES
```
Note: I am having huge performance issues when running the training on a docker so unless you run it on the cloud you may have to train the model without docker (see 3.)
## 6.2 Flask
### 6.2.1 Build
```console
make docker_build_flask
```
or with the full command
```console
docker build \
    -t flask:v0 \
    -f docker/flask/Dockerfile .
```
### 6.2.2 Run
```console
make docker_run_flask
```
or with the full command
```console
docker run \
    -p $FLASK_PORT:$FLASK_PORT \
    -v $(PWD)/models:/anime/models \
    flask:v0 \
        --model_path=./models/trained_model.pkl \
        --debug False \
        --host_ip $LOCAL_IP_ADRESS \
        --port $FLASK_PORT
```
Note: Same as for the training, I am having performance issues when running the training on a docker locally so I would advise you to run flask without docker (see 4.).

## 6.3 Streamlit
### 6.2.1 Build
```console
make docker_build_streamlit
```
or with the full command
```console
docker build \
    -t streamlit:v0 \
    -f docker/streamlit/Dockerfile .
```
### 6.2.2 Run
```console
make docker_run_streamlit
```
or with the full command
```console
docker run \
    -p $STREAMLIT_PORT:$STREAMLIT_PORT \
    -v $(PWD)/data:/anime/data \
    streamlit:v0 \
        --browser.serverAddress $LOCAL_IP_ADRESS \
        --server.port $STREAMLIT_PORT \
        -- \
            --anime_path ./data/anime.csv \
            --model_ip $LOCAL_IP_ADRESS \
            --model_port $FLASK_PORT
```
For the streamlit app the performances are not harmed at all by the use of docker.

# To improve
If I were to continue this project, I would:
- Find a way to make the docker containers more efficient
- Add command lines to train and expose the model/app on the cloud
- Test Poetry to handle requirements
- Bonus: Refacto the code
