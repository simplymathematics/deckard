# Install gcloud

First you need to install `gcloud-cli`. In order to setup the GKE (Google Kubernetes Engine), we need to enable require apis. Follow [these intsructions to enable them](https://cloud.google.com/endpoints/docs/openapi/enable-api).
[Source](https://cloud.google.com/sdk/docs/install)
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
```
