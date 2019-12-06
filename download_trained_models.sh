#!/bin/bash

if [ ! -d "Models" ]; then
	mkdir Models
fi

wget -bqco download_team8_uzeros_15epoch.log "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uzeros_15epoch.pt" -O Models/model_team8_uzeros_15epoch.pt
wget -bqco download_team8_uzeros.log "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uzeros.pt" -O Models/model_team8_uzeros.pt
wget -bqco download_team8_uzeros.log "https://ml2-final-project-data.s3.amazonaws.com/model_team8_uones.pt" -O Models/model_team8_uones.pt


echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
echo "wget output is being directed to download_stanford_data.log"
echo "use 'tail -f download_standford_data.log' to check progress"
