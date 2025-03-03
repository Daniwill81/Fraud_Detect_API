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

    PROJ_NAME: str = "sfd-api"

    # Envs
    APP_ENV: str = os.getenv("APP_ENV", "DEV")
    LOG_DIR: str = "/tmp/"
    APP_DIR: pathlib.Path = pathlib.Path(__file__).parent.parent
    FRONTEND_URL: str = "http://localhost:8000/"

    # Databases
    MONGO: DatabaseParams = Field(default_factory=DatabaseParams)

    # Email

    # Other
    TEST_ENV: str = os.getenv("TEST_ENV", "NONE")

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
