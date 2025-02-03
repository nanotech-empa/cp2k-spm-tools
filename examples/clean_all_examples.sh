#!/bin/bash

for dir in */; do
  # Check if the "out" folder exists and delete it
  if [ -d "${dir}out" ]; then
    echo "Deleting ${dir}out"
    rm -rf "${dir}out"
  fi
  # Check if the "cubes" folder exists and delete it
  if [ -d "${dir}cubes" ]; then
    echo "Deleting ${dir}cubes"
    rm -rf "${dir}cubes"
  fi
done