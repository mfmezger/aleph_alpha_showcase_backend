<h1>FastAPI Backend for Object Detection, Speech to Text and Aleph Alpha Connection</h1>
This is a Backend POC for a Backend that allows to use Object Detection on Videos and Images using the HuggingFace Transformers library. Aditionally this Backend connects to the Aleph Alpha AI. The Backend is written in Python using the FastAPI Framework.

## Installation of Dependencies
Conda/Miniconda is recommended. This Project uses poetry for dependency management. To install poetry run:

```pip install poetry```

To install the dependencies run:

```poetry install```

To activate your poetry Venv run:

```poetry shell```



## Use the API
To start the API in dev Mode use:
```uvicorn main:app --reload```

## API Docs
```localhost:8000/docs```

## Aleph Alpha Token
To use the Aleph Alpha AI you need to get a Token from the Aleph Alpha Team. The Token needs to be added to the .env file. The simplest way ist to dupicate the template.env and rename it to .env. Then add the Token to the .env file and you are good to go.


## Initialize Pre Commit
Necessary for Development

```pip install pre-commit```

```pre-commit install```


# How to use the API

## Using OpenAPI GUI
After installation one can start the server and use the OpenAPI GUI to test the API. The GUI is available at ```localhost:8000/docs```. The GUI allows to test the API and to see the API documentation.

If you open a Request press on try it out to test the service with your own data.
![fastapi](ressources/fastapi.png?raw=true "FastAPI Doc GUI")

After Selecting a Image you can press on execute to get the result.
![Request](ressources/execute.png?raw=true "Execution of Request")
The result is a UUID that can be used to get the video and detections with other requests. (Important: you need to remove the "" from the UUID)

![results](ressources/result.png?raw=true "Results")
