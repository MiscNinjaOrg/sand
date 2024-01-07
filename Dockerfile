# 
FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# ADD .env.local .env

WORKDIR /code/app

EXPOSE 8000

#
CMD ["python3", "main.py"]