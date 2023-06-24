terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
      version = "4.51.0"
    }
  }
}

provider "google" {
  credentials = file(var.credentials_file)

  project = var.project
  region  = var.region
  zone    = var.zone
}

resource "google_compute_network" "network" {
  name = "decard-network"
}

resource "google_compute_subnetwork" "subnetwork" {
  name          = "decard-subnetwork"
  ip_cidr_range = "172.16.0.0/28"
  region        = var.region
  network       = google_compute_network.network.id
}

resource "google_compute_firewall" "network-fw-1" {
  name    = "firewall-rules-for-k8s-api"
  network = google_compute_network.network.name

  allow {
    protocol = "tcp"
    ports    = ["6443"]
  }

  direction = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "network-fw-2" {
  name    = "firewall-rules-for-k8s-ssh"
  network = google_compute_network.network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  direction = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "network-fw-3" {
  name    = "firewall-rules-for-k8s-prometheus"
  network = google_compute_network.network.name

  allow {
    protocol = "tcp"
    ports    = ["30090", "30091"]
  }

  direction = "INGRESS"
  source_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "network-fw-4" {
  name    = "firewall-rules-for-k8s-cilium"
  network = google_compute_network.network.name

  allow {
    protocol = "tcp"
    ports    = ["8472"]
  }

  direction = "INGRESS"
  source_ranges = ["172.16.0.0/28"]
}


