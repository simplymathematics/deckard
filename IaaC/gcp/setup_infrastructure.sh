#!/bin/bash

# Load the config params
GCP_REGION=$(yq '.gcp_region' IaaC/gcp/config.yaml)
GCP_ZONE=$(yq '.gcp_zone' IaaC/gcp/config.yaml)
GCP_MACHINE_TYPE=$(yq '.gcp_machine_type' IaaC/gcp/config.yaml)
GCP_IMAGE_PROJECT=$(yq '.gcp_image_project' IaaC/gcp/config.yaml)
GCP_IMAGE_FAMILY=$(yq '.gcp_image_family' IaaC/gcp/config.yaml)
GCP_SSH_USER=$(yq '.gcp_ssh_user' IaaC/gcp/config.yaml)
GCP_SSH_PUB_KEY_FILE=$(yq '.gcp_ssh_pub_key_file' IaaC/gcp/config.yaml)
NODE_COUNT=$(yq '.node_count' IaaC/gcp/config.yaml)
read -d $'\x04' PUBLIC_KEY < $GCP_SSH_PUB_KEY_FILE

function is_gcloud_installed {
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud command not found. Please install the Google Cloud SDK."
        exit 1
    fi
}

function is_logged_in {
    if gcloud auth list --format="value(account)" | grep -q .; then
        echo "You have already logged in into the GCP."
    else
        echo "Pleas log in to the GCP using the following command:"
        echo "\t gcloud auth login"
        exit 1
    fi        
}

is_gcloud_installed
is_logged_in

# Get the currently selected project
selected_project=$(gcloud config get-value project)

echo "At the moment the $selected_project is selected as your default project in GCP."
echo -n "Is it the right one? [Y/n]:"
read -r right_project


case $right_project in
    [nN]) echo "Please select the right project as your default project."
    exit 1;;
esac

# Create GCP Service Account
SA_NAME="deckard-terraform-sa"
SA_EMAIL=${SA_NAME}@${selected_project}.iam.gserviceaccount.com
if gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
    echo Service account $SA_NAME already exists.
else
    gcloud iam service-accounts create $SA_NAME  --description "Service Account for Deckard Terraform Infrastructure" --display-name "$SA_NAME" --quiet
    gcloud projects add-iam-policy-binding $selected_project --member serviceAccount:${SA_NAME}@${selected_project}.iam.gserviceaccount.com --role roles/editor --no-user-output-enabled --quiet

    SA_KEY_FILE="./IaaC/gcp/tf-service-account.json"
    gcloud iam service-accounts keys create ${SA_KEY_FILE} --iam-account ${SA_NAME}@${selected_project}.iam.gserviceaccount.com --quiet
fi


# Create GCP Network and Subnets
gcloud compute networks create k8s-cluster --subnet-mode custom --quiet
gcloud compute networks subnets create k8s-subnet \
  --network k8s-cluster \
  --range 172.16.0.0/28 \
  --region $GCP_REGION \
  --quiet


# Create GCP firewall rules
gcloud compute firewall-rules create k8s-allow-internal \
  --allow tcp,udp,icmp \
  --network k8s-cluster \
  --source-ranges 172.16.0.0/28 \
  --quiet
gcloud compute firewall-rules create k8s-allow-external \
  --allow tcp:80,tcp:6443,tcp:443,tcp:22,icmp \
  --network k8s-cluster \
  --source-ranges 0.0.0.0/0 \
  --quiet




gcloud compute instances create k8s-master \
    --async \
    --metadata=ssh-keys="$GCP_SSH_USER:$PUBLIC_KEY" \
    --zone=$GCP_ZONE \
    --can-ip-forward \
    --image-family=$GCP_IMAGE_FAMILY \
    --image-project=$GCP_IMAGE_PROJECT \
    --machine-type=$GCP_MACHINE_TYPE \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet=k8s-subnet \
    --tags=k8s-cluster,controller \
    --quiet

gcloud compute instances bulk create \
    --async \
    --name-pattern=k8s-worker-# \
    --metadata=ssh-keys="$GCP_SSH_USER:$PUBLIC_KEY" \
    --zone=$GCP_ZONE \
    --count=$NODE_COUNT \
    --can-ip-forward \
    --image-family=$GCP_IMAGE_FAMILY \
    --image-project=$GCP_IMAGE_PROJECT \
    --machine-type=$GCP_MACHINE_TYPE \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet=k8s-subnet \
    --tags=k8s-cluster,worker \
    --quiet
