#!/bin/sh

echo "configuring kaggle..."

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
rm -r kaggle.json

echo "done configuring kaggle.\n"