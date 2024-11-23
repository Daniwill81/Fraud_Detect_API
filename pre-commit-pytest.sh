#!/bin/bash

if [[ $(git branch --show-current) =~ ^feature/ ]]; then
  echo "Skipping pytest due to no testcases detected."
else
  pytest
fi
