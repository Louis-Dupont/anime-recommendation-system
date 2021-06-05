FROM python:3.8

WORKDIR /anime
RUN mkdir data

# RUN cd ../../.
# RUN echo $(ls -1)
RUN cd ../../.
COPY requirements.txt .
COPY setup.py .

RUN mkdir src

RUN pip install -r requirements.txt

COPY src/* src/.

# RUN make train
ENTRYPOINT [ "python", "src/training.py" ]
