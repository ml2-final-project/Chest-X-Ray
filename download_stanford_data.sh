#!/bin/bash

if [ ! -d "Data" ]; then
	mkdir Data
fi

wget -bqco download_stanford_data.log "https://ml2-final-project-data.s3.amazonaws.com/CheXpert-v1.0-small.zip" -O Data/CheXpert-v1.0-small.zip

