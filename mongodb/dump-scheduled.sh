# TODO Ed, dump only specific databases
CURR_TIME=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
mongodump --archive=./data/dumps/CURR_TIME.gz --gzip