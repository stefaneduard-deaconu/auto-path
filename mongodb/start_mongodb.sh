source .env

# remove older mongodb container and image:

CONTAINER_NAME="mongodb"
OLD_CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
OLD_IMAGE_ID=$(sudo docker images --filter=reference="$CONTAINER_NAME" --quiet)

if [ ! -z "$OLD_CONTAINER_ID" ]
then
    echo "Stopping and removing container $OLD_CONTAINER_ID"
    docker exec -it $OLD_CONTAINER_ID kill 1
    sleep 1
    docker rm $OLD_CONTAINER_ID --force
fi
if [ ! -z "$OLD_IMAGE_ID" ]
then
    echo "Removing image $OLD_IMAGE_ID"
    sudo docker rmi $OLD_IMAGE_ID --force
fi

# THIS CAN BE USED FOR STARTING A MONGO DB AND CAN BE ACCESSED REMOTELY
sudo docker run \
    -d -p 27017:27017 \
    -v "$MONGO_VOLUME_PATH:/data/db" \
    -v "$MONGO_DUMPS_PATH:/data/dumps" \
    --name mongodb arm64v8/mongo:3.7.9-xenial

# https://www.baeldung.com/ops/docker-cron-job
# sleep 1
# echo "Add automatic hourly dumping"
# NEW_CONTAINER_ID=$(docker ps -aqf "name=$CONTAINER_NAME")
# docker cp dump-scheduled.sh $NEW_CONTAINER_ID:/bin
# docker exec $NEW_CONTAINER_ID bash /bin/dump-scheduled.sh  