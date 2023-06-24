variable "credentials_file" {
    default = "tf-service-account.json"
}

variable "project" { }

variable "region" {
  default = "us-central1"
}

variable "zone" {
  default = "us-central1-c"
}

variable "machine_type" {
  default = "e2-medium"
}

variable "image" {
  default = "ubuntu-os-cloud/ubuntu-2204-lts"
}

variable "gce_ssh_user" {
  default = "mr.salehsedghpour"
}

variable "gce_ssh_pub_key_file" {
  default = "~/.ssh/id_rsa.pub"
}

variable "node_count" {
  default = "3"
}
