#!/bin/bash

if [[ $(git branch --show-current) =~ ^feature/ ]]; then
  echo "Skipping pytest due to draft branch detected."
else
  pytest
fi
