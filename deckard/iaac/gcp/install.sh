#!/bin/bash
cd ~
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-444.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-444.0.0-linux-x86.tar.gz
bash ./google-cloud-sdk/install.sh
cd - 
