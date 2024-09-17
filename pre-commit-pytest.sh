#!/bin/bash

if [[ $(git branch --show-current) =~ ^draft/ ]]; then
  echo "Skipping pytest due to draft branch detected."
else
  pytest
fi
