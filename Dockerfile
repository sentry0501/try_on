FROM python:3.8

# EXPOSE 8000

WORKDIR try_on_docker

ADD . /try_on_docker
# install dependencies
# install project requirements

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install -U pip
RUN pip3 install -r requirements.txt
 
# Run app.py when the container launches
CMD [ "python", "main.py"]
