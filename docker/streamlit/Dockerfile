FROM python:3.7

WORKDIR /anime

RUN cd ../../.
COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt

#RUN mkdir streamlit
RUN mkdir data
RUN mkdir src
COPY src/* src/.
COPY streamlit/. .

ENTRYPOINT ["streamlit", "run", "app.py"]