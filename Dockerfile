FROM tensorflow/tensorflow:2.10.0

WORKDIR /app
COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

CMD ["uwsgi", "--ini", "app.ini"]
