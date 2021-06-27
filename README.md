PROJECT_NAME
==============================

DESC

Project Organization
------------

    ├── docker             <- folder with all the useful Dockerfiles
    │
    ├── flask              <- Flask API used to serve the model
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── src                <- Source code for use in this project.
    │
    ├── streamlit          <- Streamlit user interface 
    │
    ├── Makefile           <- Makefile with commands like `make train`
    │
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# How to run the project
## 1. Environment Initialization
The variables used in this ReadMe are all defined in the **.env** file.<br/>
These variables are automatically imported in the Makefile but if you want to run any of the following command manually you have to run either of these command lines
```console
make init_env_variables
```
or with the full command
```console
export $(grep -v '^#' .env | xargs)
```

## 2. Get the data
The data used in this project XXXXX
Store it into the data folder

## 3. Train the model
```console
make train
```
or with the full command
```console
python src/training.py \
    --data_folder $(PWD)/data \
    --model_path $(PWD)/models/trained_model.pkl \
    --nb_hidden_features $(NB_FEATURES)
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
    --model_path $(PWD)/models/trained_model.pkl \
    --debug False \
    --host_ip $FLASK_IP \
    --port $FLASK_PORT
```
The server should be ready to use in a few seconds
### 4.2 Test it (optional)
```console
make predict_shingeki_no_kyojin
```
or with the full command
```console
curl -X POST http://$FLASK_IP:$FLASK_PORT/predict \
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
    --server.port $STREAMLIT_PORT \
    -- \
        --model_ip $FLASK_IP \
        --model_port $FLASK_PORT
```
### 5.2 Try it out!
If not tab started on your brower you can run this command and paste the result in your favorite browser
```console
echo http://$STREAMLIT_IP:$STREAMLIT_PORT
```

# Docker
<p>To add memory to the container: https://stackoverflow.com/questions/44533319/how-to-assign-more-memory-to-docker-container</p>