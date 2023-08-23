# Required info for GCP

Some variables should be defined in `variables.tf` file. For instance:

- `region`
- `zone`
- `machine_type`
- `image`
- `gce_ssh_user`
- `gce_ssh_pub_key_file`
- `node_count`


> **_NOTE:_**  For now, it only creates the required infrastructure. The next step is to install and configure K8s. 
