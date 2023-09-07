import os
import subprocess
import logging
from dataclasses import dataclass
from pathlib import Path

# import yaml


logger = logging.getLogger(__name__)

__all__ = ["GCP_Config"]

logging.basicConfig(level=logging.INFO)
secret_file = os.environ.get("GCP_SECRET_FILE")
assert secret_file is not None, "Please set the GCP_SECRET_FILE environment variable"
secret_file = Path(secret_file).resolve()
project_name = os.environ.get("GCP_PROJECT_NAME")
assert project_name is not None, "Please set the GCP_PROJECT_NAME environment variable"
assert Path(secret_file).exists(), f"File {secret_file} does not exist"
# Login to GCP

try:
    command = [
        "gcloud",
        "auth",
        "login",
        f"--cred-file={secret_file}",
        f"--project={project_name}",
    ]
    logger.info(f"Running command: {command}")
    output = subprocess.run(command)
    logger.info(f"{output}")
except:  # noqa: E722
    raise ImportError(
        "Error logging in to GCP. Please check your credentials and ensure that gcloud cli is installed.",
    )


@dataclass
class GCP_Config:
    num_nodes: int = 1
    cluster_name: str = "k8s-cluster"
    gpu_type: str = "nvidia-tesla-v100"
    gpu_count: int = 1
    gpu_driver_version: str = "default"
    machine_type: str = "n1-standard-2"
    min_nodes: int = 1
    max_nodes: int = 1
    conf_dir: str = "./conf/gcp/"
    storage_config: str = "sclass.yaml"
    persistent_volume_claim: str = "pvc.yaml"
    pod: str = "pod.yaml"
    image_project: str = "ubuntu-os-cloud"
    image_family: str = "ubuntu-2204-lts"
    mount_directory: str = "/mnt/filestore"
    _target_: str = "deckard.gcp.deploy.GCP_Config"
    region: str = "europe-west4"

    def create_cluster(self):
        # Create a cluster
        logger.info(
            f"Creating cluster {self.cluster_name} in region {self.region} with {self.num_nodes} nodes",
        )
        command = f"gcloud container clusters create {self.cluster_name}  --num-nodes {self.num_nodes} --no-enable-autoupgrade --addons=GcpFilestoreCsiDriver"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def install_kubectl(self):
        logger.info("Installing kubectl on the local machine")
        command = "gcloud components install kubectl"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def retrieve_credentials(self):
        logger.info(
            f"Retrieving credentials for cluster {self.cluster_name} in region {self.region}",
        )
        command = f"gcloud container clusters get-credentials {self.cluster_name}  "
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def create_node_pool(self):
        logger.info(f"Creating node pool {self.cluster_name} in region {self.region}")
        if self.gpu_type is not None:
            assert (
                self.gpu_count > 0
            ), f"Please specify a valid GPU count. Current value: {self.gpu_count}"
            command = f"gcloud container node-pools create {self.cluster_name} --accelerator type={self.gpu_type},count={self.gpu_count},gpu-driver-version={self.gpu_driver_version}  --cluster {self.cluster_name} --machine-type {self.machine_type} --num-nodes {self.num_nodes} --min-nodes {self.min_nodes} --max-nodes {self.max_nodes}"
        else:
            command = f"gcloud container node-pools create {self.cluster_name}  --cluster {self.cluster_name} --machine-type {self.machine_type} --num-nodes {self.num_nodes} --min-nodes {self.min_nodes} --max-nodes {self.max_nodes}"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"{output}")
        return output.stdout

    def create_deployment(self):
        logger.info(f"Creating deployment {self.cluster_name} in region {self.region}")
        file_name = Path(self.conf_dir, self.storage_config).resolve().as_posix()
        command = f"kubectl create -f {file_name}"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def create_persistent_volume_claim(self):
        logger.info(
            f"Creating persistent volume claim {self.cluster_name} in region {self.region}",
        )
        file_name = (
            Path(self.conf_dir, self.persistent_volume_claim).resolve().as_posix()
        )
        command = f"kubectl create -f {file_name}"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def deploy_pod(self):
        logger.info(f"Creating sample pod {self.cluster_name} in region {self.region}")
        file_name = Path(self.conf_dir, self.pod).resolve().as_posix()
        command = f"kubectl create -f {file_name}"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(command)
        logger.info(f"{output}")
        return output

    def prepare_access_values(self):
        logger.info(
            f"Preparing access values in the shared volumee {self.cluster_name} in region {self.region}",
        )
        command = f"gcloud compute instances create filestore --async --image-family={self.image_family} --image-project={self.image_project} --machine-type={self.machine_type} --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring --subnet=default --quiet"
        logger.info(f"Running command: {command}")
        command = command.split(" ")
        output = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"{output}")
        return output.stdout
    def __call__(self):
        self.create_cluster()
        self.install_kubectl()
        self.retrieve_credentials()
        self.create_node_pool()
        self.create_deployment()
        self.create_persistent_volume_claim()
        self.deploy_pod()
        self.prepare_access_values()
        return None
