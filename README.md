[![Release 0.0.1](https://github.com/trellixio/proj-ehadj-api)]
[![Python 3.12](https://img.shields.io/badge/python-3.12-yellow.svg)](https://www.python.org/downloads/release/python-312/)
[![MongoDB 7.0](https://img.shields.io/badge/mongodb-7.0-green.svg)](https://www.mongodb.com/docs/v7.0/)


# efraud dectect Project


This is the app project template structure using FastAPI.
This template is based on the following technologies:
- [FastAPI](https://fastapi.tiangolo.com/) as web framework
- [MongoDB](https://www.mongodb.com/docs/) as database
- [Beanie](https://roman-right.github.io/beanie/) as ODM
- [Pytest](https://docs.pytest.org/) as testing framework
- [Github Actions](https://docs.github.com/en/actions) for workflow automation


## 🔨 Installing

### Python 3.9
Before getting started, ensure that [Python 3.12](https://www.python.org/) is installed on your computer.

### MongoDB
MongoDB is a document-oriented database that use JSON-like documents to store data.
You will need to install MongoDB 6.0 [locally](https://www.mongodb.com/docs/manual/installation/)
or sign up for a free hosted one with [MongoDB Atlas](https://www.mongodb.com/pricing).
Once you install MongoDB, make sure to create a database.

### Steps
Clone the repo and open a terminal at the root of the cloned repo.

Before getting started, make sure you have installed virtualenv on your device:

Install virtualenv
```shell
sudo apt install python3-virtualenv
```

1. Setup a virtual env. Only do this on your first run.
```shell
python3.12 -m venv .venv
```

2. Activate the virtualenv
```shell
source .venv/bin/activate
```

3. Install all dependencies:
```shell
pip install -r requirements-dev.txt
```

4. Set up pre-commit
```shell
pre-commit install
```

5. Init environment variables. Duplicate the env template file:
```shell
cp ./.env.tpl ./.env
```
Open `.env` file with a text editor and update the env vars as needed


6. Load metadata, from metadata.yml:
```shell
python register.py
```

1. Run migrations:
```shell
python migrate.py
```

## ⏯ Running

To run the app on your machine, make sure to have activated your virtualenv and run:
```shell
python main.py
```

- Visit: http://127.0.0.1:8000/ to view the application running in your browser


## 🖌 Formatting

Keep in mind that those are automated formatting assistant tools.
They will not always give the best result, as they just apply
rules blindly. As a developer you still have the responsibility to
ensure that the code is formatted with perfection.

- Use black to format the code
From the project root run:
```shell
black .
```

- Use isort to sort the import
From the project root run:
```shell
isort .
```


## 🧽 Linting

Linters are useful to ensure that your code quality matches with standards.

- Running pre-commit on the project to run all linters.
```shell
pre-commit run --all-files
```

- Use pylint to check for common mistakes.
From the project root, run:
```shell
pylint AppMain app tests
```

- Use mypy to check for typing issues.
From the project root, run:
```shell
mypy .
```

- Use pydocstyle to check for documentation issues.
From the project root, run:
```shell
pydocstyle .
```


## 🧪 Testing

Tests are run using the pytest library.
From the project root, run:
```shell
pytest
```

To test a specific testcase:
```shell
pytest -sv path_to_test_file::test_name
```
