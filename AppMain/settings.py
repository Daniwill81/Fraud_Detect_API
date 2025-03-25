"""
Application Settings.

The settings file contains all the configuration of the application.
Use this file to configure any parameters that should be set server-side.

Settings can be adjusted depending on the environment the app is running on.
The environment is usually defined by the `APP_ENV` OS environment variable that
can be set on the local machine or on the server.

!!! warning !!!
For security reason, do not put in this file any secret key.
Add them as OS environment variables or put them in the `.env` file.

https://github.com/theskumar/python-dotenv

"""

import locale
import logging
import logging.config
import os
import pathlib
import typing

import aioboto3
import pydantic_settings
from pydantic import Field

from sap.settings import DatabaseParams

locale.setlocale(locale.LC_ALL, "")


class _Settings(pydantic_settings.BaseSettings):
    """
    Application Settings.

    The setting are load from environment variables:
    https://pydantic-docs.helpmanual.io/usage/settings/

    All env variable should be prefixed with APP_SETTINGS_
    For example to set the LOG_DIR, use: APP_SETTINGS_LOG_DIR="/tmp/"
    """

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=os.getenv("APP_DOTENV", ".env"),
        env_file_encoding="utf-8",
        env_prefix="APP_SETTINGS_",
        env_nested_delimiter="__",
    )

    PROJ_NAME: str = "eFraude-detection"

    # Envs
    APP_ENV: str = os.getenv("APP_ENV", "DEV")
    LOG_DIR: str = "/tmp/"
    APP_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent
    FRONTEND_URL: str = ""

    # Databases
    MONGO: DatabaseParams = Field(default_factory=DatabaseParams)

    # Files Storage
    AWS_ENDPOINT: str
    AWS_ACCESS_KEY_ID: str
    AWS_ACCESS_KEY_SECRET: str
    AWS_REGION: str = "eu-north-1"
    AWS_S3_BUCKET: str = "e-fraud-demo-bucket-1"
    AWS_USE_SSL: bool = True

    # SageMaker endpoints
    SAGEMAKER_STANDARD_ENDPOINT: str = "https://your-standard-sagemaker-endpoint.amazonaws.com"
    SAGEMAKER_CKKS_ENDPOINT: str = "https://your-ckks-sagemaker-endpoint.amazonaws.com"

    # Model settings
    FRAUD_THRESHOLD: float = 0.5  # Threshold to determine if transaction is fraudulent

    @property
    def is_prod(self) -> bool:
        """Return True if production environment."""
        return self.APP_ENV == "PROD"

    @property
    def is_dev(self) -> bool:
        """Return True if dev environment."""
        return self.APP_ENV == "DEV"


AppSettings = _Settings()


# ###################################
# #     Logging       ###############
# ###################################


def logging_setter() -> dict[str, typing.Any]:
    """Set the logging config for the app."""
    log_dir: str = AppSettings.LOG_DIR
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "root": {
            "level": "WARNING",
            "handlers": ["console", "file"],
        },
        "formatters": {
            "verbose": {"format": "%(levelname)s %(asctime)s %(module)s %(process)d %(thread)d %(message)s"},
            "simple": {"format": "%(asctime)s %(levelname)s %(message)s"},
        },
        "handlers": {
            "null": {
                "level": "DEBUG",
                "class": "logging.NullHandler",
            },
            "console": {
                "level": "DEBUG",
                "class": "logging.StreamHandler",
                "formatter": "verbose",
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": os.path.join(log_dir, "app.log"),
                "formatter": "simple",
            },
            "file_access": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": os.path.join(log_dir, "access.log"),
                "formatter": "simple",
            },
        },
        "loggers": {
            "app": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "access": {
                "level": "DEBUG",
                "handlers": ["file_access"],
                "propagate": False,
            },
            "sap": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }


logging.config.dictConfig(logging_setter())

logger = logging.getLogger("app")
logger_access = logging.getLogger("access")


# ###################################
# #     Logging       ###############
# ###################################

boto3_session = aioboto3.Session(
    aws_access_key_id=AppSettings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AppSettings.AWS_ACCESS_KEY_SECRET,
    region_name=AppSettings.AWS_REGION,
)
