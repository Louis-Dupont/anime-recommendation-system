FROM python:3.8

WORKDIR /anime

RUN cd ../../.
COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt

RUN mkdir models
RUN mkdir src
RUN mkdir flask
COPY src/* src/.
COPY flask/* flask/.

# RUN make train
ENTRYPOINT [ "python", "flask/app.py"]
