#!/bin/bash

wget -bqco download_stanford_data.log "https://us13.mailchimp.com/mctx/click?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0-small.zip&xid=3a61aaac7b&uid=55365305&pool=&subject=" -O CheXpert-v1.0-small.zip

echo "Download is proceeding in the background."
echo "Should a problem occur, rerunning this script, will resume the download."
echo "wget output is being directed to download_stanford_data.log"
echo "use 'tail -f download_standford_data.log' to check progress"
