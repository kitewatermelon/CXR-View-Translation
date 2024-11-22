docker build -t docker-python:latest .
docker rm docker-python
docker run --gpus all -it --name docker-python docker-python:latest