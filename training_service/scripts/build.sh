#!/bin/bash

echo "Installing requirements"
pip install -r requirements.txt

echo "Migrating database"
python migrate.py
