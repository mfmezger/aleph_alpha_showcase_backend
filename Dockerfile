FROM python:3.9

WORKDIR /app
ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PORT=8000

# Install Poetry
RUN curl -sSL https://install.python-poetry.org/ | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* /app/

RUN apt-get update && apt-get install -y python3-opencv ffmpeg
RUN poetry install --no-root --no-dev

# installing the timm libary
RUN pip install timm

# moving the complete app as well as the stored models into the docker workspace
COPY /backend /app

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# ENTRYPOINT ["tail", "-f", "/dev/null"]
