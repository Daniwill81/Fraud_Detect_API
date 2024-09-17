#!/bin/bash

echo "Starting uvicorn"
uvicorn AppMain.asgi:app --host 0.0.0.0 --port 10000 --proxy-headers --forwarded-allow-ips "*"
