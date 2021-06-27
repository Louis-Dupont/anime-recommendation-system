FROM python:3.7

WORKDIR /anime

COPY docker/streamlit/requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt

#RUN mkdir streamlit
COPY streamlit/. .
COPY test.py .

ENTRYPOINT ["streamlit", "run", "app.py"]