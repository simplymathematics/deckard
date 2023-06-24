#!/bin/bash

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
    gcloud iam service-accounts create $SA_NAME  --description "Service Account for Deckard Terraform Infrastructure" --display-name "$SA_NAME"
    gcloud projects add-iam-policy-binding $selected_project --member serviceAccount:${SA_NAME}@${selected_project}.iam.gserviceaccount.com --role roles/editor --no-user-output-enabled

    SA_KEY_FILE="./IaaC/gcp/tf-service-account.json"
    gcloud iam service-accounts keys create ${SA_KEY_FILE} --iam-account ${SA_NAME}@${selected_project}.iam.gserviceaccount.com
fi

# execute terraform
terraform -chdir=./IaaC/gcp/ init -var="project=$selected_project"
terraform -chdir=./IaaC/gcp/ plan -var="project=$selected_project"
terraform -chdir=./IaaC/gcp/ apply -var="project=$selected_project"




