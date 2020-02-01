FROM tiangolo/uwsgi-nginx-flask:python3.7

RUN pip install --upgrade pip && \
    pip install \
        torch \
        torchvision \
        numpy \
        matplotlib


COPY . /app