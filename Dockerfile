FROM python:3.8-slim-buster
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
WORKDIR /app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
CMD [ "python", "app.py", "--host=0.0.0.0"]