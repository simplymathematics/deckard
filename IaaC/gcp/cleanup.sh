#!/bin/bash

selected_project=$(gcloud config get-value project)
SA_NAME="deckard-terraform-sa"


terraform -chdir=./IaaC/gcp/ destroy -var="project=$selected_project"

gcloud iam service-accounts delete ${SA_NAME}@${selected_project}.iam.gserviceaccount.com --quiet

rm -rf ./IaaC/gcp/.terraform ./IaaC/gcp/.terraform.lock.hcl ./IaaC/gcp/terraform.tfstate ./IaaC/gcp/terraform.tfstate.backup ./IaaC/gcp/tf-service-account.json



