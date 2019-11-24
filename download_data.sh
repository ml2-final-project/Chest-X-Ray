#!/bin/bash

# assumes you have got the kaggle.json file from kaggle, using these instructions:
# https://github.com/Kaggle/kaggle-api

# put the file under ~/.kaggle/kaggle.json
if [ ! -f /home/${USER}/.kaggle/kaggle.json ]; then
	echo "kaggle.json file missing"
	exit 1
fi

pip install kaggle --user kaggle

/home/${USER}/.local/bin/kaggle datasets download  -p Data nih-chest-xrays/sample
