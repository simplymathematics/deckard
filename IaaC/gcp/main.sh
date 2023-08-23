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

IPs=( $(gcloud compute instances list | awk  '{ print $5 }' | awk '(NR>1)'  ))
for i in "${IPs[@]}"
    do
        ssh-keyscan -H $i >> ~/.ssh/known_hosts
    done

# Create inventory file
echo '[master]' > IaaC/ansible_files/inventory.ini
gcloud compute instances list  | grep master | awk '{ print $5 }' >> IaaC/ansible_files/inventory.ini

echo  -e "\n[workers]" >> IaaC/ansible_files/inventory.ini
gcloud compute instances list  | grep worker | awk '{ print $5 }' >> IaaC/ansible_files/inventory.ini


