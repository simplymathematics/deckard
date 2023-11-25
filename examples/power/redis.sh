#!/bin/bash
sudo apt install lsb-release curl gpg -y
ls /usr/share/keyrings/redis-archive-keyring.gpg || curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg 
echo "deb [trusted=yes] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update -y
sudo apt-get install redis -y
if [ ! -f .bashrc ]; then
    if [[ ! -z "$REDIS_PORT" ]]; then
        echo "Please choose a port. Default is 6379."
        read REDIS_PORT
        echo "\$REDIS_PORT set as env var, \$REDIS_PORT."
        echo "alias REDIS_PORT=$REDIS_PORT" >> .bashrc
    else
        echo "alias REDIS_PORT=$REDIS_PORT" >> .bashrc
    fi
    if [[ ! -z "$REDIS_PASSWORD" ]]; then 
        echo "Please set a password"
        read REDIS_PASSWORD
        echo "alias REDIS_PASSWORD=$REDIS_PASSWORD" >> .bashrc
    else
        echo "alias REDIS_PASSWORD=$REDIS_PASSWORD" >> .bashrc
    fi
fi
source .bashrc