#!/bin/bash

# Install Dependencies
apt install lsb-release curl gpg -y
# Add GPG Key
curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
# Add source to sources list
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list
# Update
apt-get update -y
# Install
apt-get install redis -y
# Copy Default Config
cp /usr/local/etc/redis.conf.default /usr/local/etc/redis.conf
# Make db folder
mkdir -p /usr/local/var/db/redis
# # Launch 
# redis-server