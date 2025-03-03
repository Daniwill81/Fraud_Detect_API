# pylint: disable=unused-import

"""
# Main.

This is the main entrypoint used by deta.sh
It just needs to reference the app object
#
#  You probably do not want to edit this file.
"""

from pprint import pprint

import uvicorn

from AppMain.asgi import app
from AppMain.settings import AppSettings

if __name__ == "__main__" and AppSettings.APP_ENV == "DEV":
    pprint(AppSettings.model_dump())
    uvicorn.run("AppMain.asgi:app", reload=True, log_level="debug")
