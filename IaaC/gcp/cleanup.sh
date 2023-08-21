#!/bin/bash

selected_project=$(gcloud config get-value project)
SA_NAME="deckard-terraform-sa"

# Load config params
GCP_REGION=$(yq '.gcp_region' IaaC/gcp/config.yaml)
GCP_ZONE=$(yq '.gcp_zone' IaaC/gcp/config.yaml)
GCP_MACHINE_TYPE=$(yq '.gcp_machine_type' IaaC/gcp/config.yaml)
GCP_IMAGE_PROJECT=$(yq '.gcp_image_project' IaaC/gcp/config.yaml)
GCP_IMAGE_FAMILY=$(yq '.gcp_image_family' IaaC/gcp/config.yaml)
GCP_SSH_USER=$(yq '.gcp_ssh_user' IaaC/gcp/config.yaml)
GCP_SSH_PUB_KEY_FILE=$(yq '.gcp_ssh_pub_key_file' IaaC/gcp/config.yaml)
NODE_COUNT=$(yq '.node_count' IaaC/gcp/config.yaml)
read -d $'\x04' PUBLIC_KEY < $GCP_SSH_PUB_KEY_FILE


# Delete instances
gcloud compute instances delete k8s-master --quiet --zone=$GCP_ZONE
for (( i=1; i <= $NODE_COUNT; i++ ))
do
  gcloud compute instances delete k8s-worker-${i}  --quiet --zone=$GCP_ZONE
done


# Delete firewall rules
gcloud compute firewall-rules  delete k8s-allow-internal --quiet
gcloud compute firewall-rules  delete k8s-allow-external --quiet

# Delete network and subnetwork
gcloud compute networks subnets delete k8s-subnet --region $GCP_REGION --quiet
gcloud compute networks delete k8s-cluster --quiet

# Delete service account
gcloud iam service-accounts delete ${SA_NAME}@${selected_project}.iam.gserviceaccount.com --quiet


rm -rf ./IaaC/gcp/.terraform ./IaaC/gcp/.terraform.lock.hcl ./IaaC/gcp/terraform.tfstate ./IaaC/gcp/terraform.tfstate.backup ./IaaC/gcp/tf-service-account.json



