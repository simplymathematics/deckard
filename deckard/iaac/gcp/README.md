# Run the experiments in GCP

The following process requires to have enough qouta in GCP for CPUs and GPUs and SSDs. The whole procedure should take around 30 minutes to complete.

## Setup the GKE + GPU

0- In order to setup the GKE (Google Kubernetes Engine), we need to enable require apis. Follow [these intsructions to enable them](https://cloud.google.com/endpoints/docs/openapi/enable-api).
## Install gcloud on your local machine

First you need to install `gcloud-cli`. In order to setup the GKE (Google Kubernetes Engine), we need to enable require apis. Follow [these intsructions to enable them](https://cloud.google.com/endpoints/docs/openapi/enable-api).
[Source](https://cloud.google.com/sdk/docs/install)
```
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl sudo
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update && sudo apt-get install google-cloud-cli
```




1-We then create the cluster called `k8s-cluster`. This cluster will be installed in `europe-west4` in GCP regions.
```
gcloud container clusters create k8s-cluster \
    --zone europe-west4-a --num-nodes 3 --no-enable-autoupgrade --addons=GcpFilestoreCsiDriver
```

2- In order to manage the Kubernetes cluster we need to [install `kubectl`](https://kubernetes.io/docs/tasks/tools/#kubectl).

3- Then run the following command to get the credentials:
```
gcloud container clusters get-credentials k8s-cluster \
    --zone europe-west4-a
```

After this step you should now be able to access kubernetes cluster, give it a try by simply running the following:
```
$ kubectl get nodes
```
The results should be similar to:
```
NAME                                          STATUS   ROLES    AGE     VERSION
gke-k8s-cluster-default-pool-1fa99288-r4cz    Ready    <none>   5m   v1.27.3-gke.100
gke-k8s-cluster-default-pool-3a09572b-w13l    Ready    <none>   5m   v1.27.3-gke.100
gke-k8s-cluster-default-pool-feae82f2-zfsj    Ready    <none>   5m   v1.27.3-gke.100
```

4- Now we want to create a node pool called `k8s-node-pool` with `nvidia-tesla-v100` gpus. Run the following.
```
gcloud container node-pools create k8s-node-pool \
  --accelerator type=nvidia-tesla-v100,count=1,gpu-driver-version=default \
  --zone europe-west4-a --cluster k8s-cluster \
  --machine-type n1-standard-2		\
  --num-nodes 1 \
   --min-nodes 1 \
   --max-nodes 1
```

After succesfully running the above command, you can verify the added GPU nodes by running `kubectl get nodes` and it should be something like:
```
NAME                                          STATUS   ROLES    AGE     VERSION
gke-k8s-cluster-default-pool-1fa99288-r4cz    Ready    <none>   10m   v1.27.3-gke.100
gke-k8s-cluster-default-pool-3a09572b-w13l    Ready    <none>   10m   v1.27.3-gke.100
gke-k8s-cluster-default-pool-feae82f2-zfsj    Ready    <none>   10m   v1.27.3-gke.100
gke-k8s-cluster-k8s-node-pool-7ba6832e-n8ld   Ready    <none>   3m   v1.27.3-gke.100
gke-k8s-cluster-k8s-node-pool-e7fca6ba-wr5l   Ready    <none>   3m   v1.27.3-gke.100
gke-k8s-cluster-k8s-node-pool-f075e2ff-zvj6   Ready    <none>   3m   v1.27.3-gke.100
```


## Preparing the storage
1- To be able to have a shared storage for our containers/pods, we need to define `storage class` by simply running
```
kubectl create -f ./IaaC/gcp/sclass.yaml
```

2- After that we need to create `persistent volume claim` to enable the pod to have access to the volume by simply ruuning:
```
kubectl create -f ./IaaC/gcp/pvc.yaml
```


## Deploying the GPU and Storage enabled pod
The pod should include the following resources to enable access to GPU:
```
...
    resources:
      limits:
       nvidia.com/gpu: 1
...
```

And it also should have the following volumes to enable the shared storage:
```
...
    volumeMounts:
        - mountPath: /data
          name: mypvc
  volumes:
  - name: mypvc
    persistentVolumeClaim:
      claimName: podpvc
...
```

You can simply take a look at `pod.yaml` file for defining a pod. Just to check, deploy the sample pod by running:
```
kubectl apply -f ./IaaC/gcp/pod.yaml
```

## Install Kepler and monitoring tools
Kepler is the module that collects the power consumption per container/namespace/node and stores them in Prometheus:
```bash
kubectl apply --server-side -f ./IaaC/gcp/prometheus/setup
kubectl apply -f ./IaaC/gcp/prometheus/
kubectl apply -f ./IaaC/gcp/kepler/deployment.yaml
```

## Prepare the access values in the shared volume (optional):
First of all we need to create a vm by running:
```
gcloud compute instances create filestore \
    --async \
    --metadata=ssh-keys="username:publickey" \
    --zone=europe-west4-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-2 \
    --scopes compute-rw,storage-ro,service-management,service-control,logging-write,monitoring \
    --subnet=default \
    --quiet
```

In the above command, you simply replace you public key and your username to have passwordless SSH to the VM.

Before sshing to the created instance, simply get the [`NFS mount point` from the GCP console](https://console.cloud.google.com/filestore/instances). The `mount point` is "`Filestore instance IP address`:`Filestore instance File share name`", for instance: `<your ip>:/vol1`.

After it has been created, run the `gcloud compute instances list` command to retrieve the external ip of the created instance `filestore`. Then ssh to the machine and mount the volume by running the followings:

```
sudo apt update
sudo apt install nfs-common
mkdir mount-directory
sudo mount -o rw,intr 10.178.118.130:/vol1 mount-directory
```

**Don't Forget to update the address in the mount command.**
