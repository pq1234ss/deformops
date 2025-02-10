#!/usr/bin/env bash
#
# List the tags of a repository in the Docker Hub that match a pattern.
#
set -e

# Input
# -----
REPO="${1:-pytorch/pytorch}"
PATTERN="${2:-^(2.[0-9\.]+-cuda[0-9\.]+-cudnn[0-9]+)-devel$}"

# Authentication
# --------------
# We fetch a read-only token for the registry.
# Without a token, we get rate-limited.
echo "Acquiring authentication token for $REPO..." >&2
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | jq -r .token)

if [[ -z "$TOKEN" ]]; then
    echo "Failed to get token." >&2
    exit 1
fi

# Read tags
# ---------
# Uses the docker API to get a list of tags for the repo.
echo "Fetching tags..." >&2
TAGS=$(curl -s -H "Authorization: Bearer $TOKEN" "https://registry-1.docker.io/v2/$REPO/tags/list" | jq -r '.tags[]')

if [[ -z "$TAGS" ]]; then
    echo "Failed to get tags for $REPO." >&2
    exit 1
fi
if [[ -n "$PATTERN" ]]; then
    echo "Filtering tags with pattern: $PATTERN" >&2
    # Use grep with Extended Regular Expressions (-E) to filter the list
    TAGS=$(echo "$TAGS" | grep -E "$PATTERN" || true)
    if [[ -z "$TAGS" ]]; then
        echo "No tags matched the provided pattern." >&2
        exit 1
    fi
fi

# Output
# ------
echo "$TAGS"
echo "Success! Found $(echo "$TAGS" | wc -l) tags." >&2
