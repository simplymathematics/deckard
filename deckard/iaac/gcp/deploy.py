import os
import subprocess
import logging
from hydra.utils import instantiate
import argparse
from dataclasses import dataclass
from pathlib import Path
import yaml


logger = logging.getLogger(__name__)

# Login to GCP
try:
    command = 'gcloud auth login'
    logger.info(f"Running command: {command}")
    output = subprocess.run(command)
    logger.info(f"{output}")
except:
    raise ImportError('Error logging in to GCP. Please check your credentials and ensure that gcloud cli is installed.')
# Get username and password from ENV
username = os.environ.get('GCP_USERNAME')
password = os.environ.get('GCP_PASSWORD')

@dataclass
class GCP_Config:
    num_nodes: int = 1
    cluster_name: str = 'k8s-cluster'
    gpu_type: str = 'nvidia-tesla-v100'
    gpu_count: int = 1
    gpu_driver_version: str = 'default'
    machine_type: str = 'n1-standard-2'
    min_nodes: int = 1
    max_nodes: int = 1
    conf_dir: str = './conf/gcp/'
    storage_config: str = 'sclass.yaml'
    persistent_volume_claim: str = 'pvc.yaml'
    pod = 'pod.yaml'
    image_project: str = 'ubuntu-os-cloud'
    image_family: str = 'ubuntu-2204-lts'
    mount_directory: str = '/mnt/filestore'
    _target_ : str = 'deckard.conf.GCP_Config'
    
    def create_cluster(self):
        # Create a cluster
        logger.info(f'Creating cluster {self.cluster_name} in region {self.region} with {self.num_nodes} nodes')
        command = f'gcloud container clusters create {self.cluster_name} --region {self.region} --num-nodes {self.num_nodes} --no-enable-autoupgrade'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def install_kubectl(self):
        logger.info(f'Installing kubectl')
        command = 'gcloud components install kubectl' 
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def retrieve_credentials(self):
        logger.info(f'Retrieving credentials for cluster {self.cluster_name} in region {self.region}')
        command = 'gcloud container clusters get-credentials {self.cluster_name} --region {self.region}'
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def create_node_pool(self):
        logger.info(f'Creating node pool {self.cluster_name} in region {self.region}')
        command = f'gcloud container node-pools create {self.cluster_name} --accelerator type={self.gpu_type},count={self.gpu_count},gpu-driver-version={self.gpu_driver_version} --region {self.region} --cluster {self.cluster_name} --machine-type {self.machine_type} --num-nodes {self.num_nodes} --min-nodes {self.min_nodes} --max-nodes {self.max_nodes}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def create_deployment(self):
        logger.info(f'Creating deployment {self.cluster_name} in region {self.region}')
        file_name = Path(self.conf_dir, self.storage_config).resolve().as_posix()
        command = f'kubectl create -f {file_name}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def create_persistent_volume_claim(self):
        logger.info(f'Creating persistent volume claim {self.cluster_name} in region {self.region}')
        file_name = Path(self.conf_dir, self.persistent_volume_claim).resolve().as_posix()
        command = f'kubectl create -f {file_name}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def deploy_pod(self):
        logger.info(f'Creating sample pod {self.cluster_name} in region {self.region}')
        file_name = Path(self.conf_dir, self.pod).resolve().as_posix()
        command = f'kubectl create -f {file_name}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def prepare_access_values(self):
        logger.info(f'Preparing access values in the shared volumee {self.cluster_name} in region {self.region}')
        command = f'gcloud compute instances create filestore --async --metadata=ssh-keys="{username}:{password}" --zone={self.region}-a --image-family={self.image_family} --image-project={self.image_project} --machine-type={self.machine_type} --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring --subnet=default --quiet'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output
    
    def find_ip_of_filestore(self):
        logger.info(f'Finding the IP address of the filestore {self.cluster_name} in region {self.region}')
        command = f'gcloud compute instances list --filter="name=filestore" --format="value(networkInterfaces[0].accessConfigs[0].natIP)" --zone={self.region}'
        logger.info(f"Running command: {command}")
        ip_output = subprocess.run(command)
        logger.info(f"{ip_output}")
        return ip_output

    def mount_filestore(self, ip):
        logger.info(f'Mounting the filestore {self.cluster_name} in region {self.region}')
        command = 'sudo apt update'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        command = 'sudo apt install nfs-common'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        command = f'mkdir {self.mount_directory}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        command = f'sudo mount -o rw,intr {ip}:/vol1 {self.mount_directory}'
        logger.info(f"Running command: {command}")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def __call__(self):
        self.create_cluster()
        self.install_kubectl()
        self.retrieve_credentials()
        self.create_node_pool()
        self.create_deployment()
        self.create_persistent_volume_claim()
        self.deploy_pod()
        self.prepare_access_values()
        ip_addr = self.find_ip_of_filestore()
        self.mount_filestore(ip_addr)


if __name__ == "__main__":
    gcp_parser = argparse.ArgumentParser()
    gcp_parser.add_argument("--verbosity", type=str, default="INFO")
    gcp_parser.add_argument("--config_dir", type=str, default="conf")
    gcp_parser.add_argument("--config_file", type=str, default="default")
    gcp_parser.add_argument("--workdir", type=str, default=".")
    args = gcp_parser.parse_args()
    config_dir = Path(args.workdir, args.config_dir).resolve().as_posix()
    config_file = Path(config_dir, args.config_file).resolve().as_posix()
    with open(config_file, "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    gcp = instantiate(params)
    gcp()