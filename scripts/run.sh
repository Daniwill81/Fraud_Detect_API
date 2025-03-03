#!/bin/bash

echo "Installing requirements"
pip install -r requirements.txt

echo "Migrating database"
python migrate.py

echo "Registering new trellis"
python register.py

echo "Starting uvicorn"
uvicorn AppMain.asgi:app --host 0.0.0.0 --port 10000 --proxy-headers --forwarded-allow-ips "*"
