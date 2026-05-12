# Docker Images and Runtime Variants

Deckard publishes and tests multiple container variants so users can choose a runtime that matches their hardware.

## Image tags

- `ghcr.io/simplymathematics/deckard:cpu`
- `ghcr.io/simplymathematics/deckard:mps`
- `ghcr.io/simplymathematics/deckard:cuda`

SHA-scoped tags are also published for each variant:

- `ghcr.io/simplymathematics/deckard:<sha>-cpu`
- `ghcr.io/simplymathematics/deckard:<sha>-mps`
- `ghcr.io/simplymathematics/deckard:<sha>-cuda`

## What each variant means

- `cpu`: Generic Ubuntu runtime, no CUDA packages.
- `mps`: Mac-oriented tag for Apple Silicon workflows. This uses the CPU container path (no CUDA) but is tagged separately for clarity.
- `cuda`: NVIDIA CUDA runtime base image and CUDA Python runtime packages enabled.

## Local builds

Build CPU image:

```bash
docker build -t deckard:cpu .
```

Build MPS-tagged image (same build path as CPU since MacOS doesn't enable hardware-acceleration using containers):

```bash
docker build -t deckard:mps .
```

Build CUDA image:

```bash
docker build \
	--build-arg ENABLE_CUDA=1 \
	--build-arg BASE_IMAGE=nvidia/cuda:12.0.0-runtime-ubuntu20.04 \
	-t deckard:cuda .
```

## CI workflows

- [docker-push.yml](../../.github/workflows/docker-push.yml) publishes `cpu`, `mps`, and `cuda` tags on pushes to `main`.
- [docker-test.yml](../../.github/workflows/docker-test.yml) builds `cpu`, `mps`, and `cuda` variants on pull requests (no publish).

For local workflow runner usage (`test_workflow.sh`), see [scripts/README.md](../../scripts/README.md).
