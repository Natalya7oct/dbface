# app/Dockerfile

FROM python:3.10.6

WORKDIR /dbface_app

COPY ./FP16 ./FP16
COPY ./dbface_lib.py ./dbface_lib.py
COPY ./app.py ./app.py
COPY ./test_video.mp4 ./test_video.mp4
COPY ./requirements.txt ./requirements.txt
COPY ./packages.txt ./packages.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
