#!/usr/bin/env nix
#! nix shell nixpkgs#bash nixpkgs#nix-prefetch-docker --command bash
#
# This script prefetches a list of tags from the Docker repository and 
# emits their definition for use in a `pkgs.dockerTools.pullImage` call.
#
IMAGE_NAME="$1"
IMAGE_TAGS=("${@:2}")

function help() {
    echo "Usage: $0 <image-name> <tag1> [<tag2> ...]"
    echo
    echo "Example: $0 pytorch/pytorch 2.0.1-cuda11.7-cudnn8-devel 2.1.0-cuda11.8-cudnn8-devel"
    echo
    echo "This will prefetch the specified tags from the given Docker image repository."
}

# Check that an image name was provided
if [ -z "$IMAGE_NAME" ]; then
    help
    exit 1
fi

# Check that tags were provided
if [ ${#IMAGE_TAGS[@]} -eq 0 ]; then
    help
    exit 1
fi

# Loop over IMAGE_TAGS, and prefetch each one
for TAG in "${IMAGE_TAGS[@]}"; do
  echo "Prefetching $IMAGE_NAME:$TAG..." >&2
  nix-prefetch-docker --image-name "$IMAGE_NAME" --image-tag "$TAG"
done


