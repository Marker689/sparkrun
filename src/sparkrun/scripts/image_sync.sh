#!/bin/bash
set -uo pipefail

# Ensure a Docker image is present on this host.
# Placeholders filled by Python: {image}, {force_pull}, {offline}
#
# NOTE: this file is consumed via Python str.format(); it must contain NO
# literal curly-brace characters.
#
# FORCE_PULL="1" is `sparkrun run --rebuild` for a registry image: skip the
# presence check and re-pull unconditionally.  The presence check is
# metadata-only -- `docker image inspect` is satisfied by an image whose
# content blobs are missing -- so it is exactly what a user reaches for
# --rebuild to bypass when the local copy is stale or incomplete.
FORCE_PULL="{force_pull}"
# OFFLINE="1" is `sparkrun run --offline`: no internet egress.  A present
# image is used as-is; an absent one is a failure (exit 3), never a pull --
# and that outranks FORCE_PULL.
OFFLINE="{offline}"
# Bare on purpose: Python substitutes an already shell-quoted value, which
# double quotes would turn back into literal quote characters around a live
# command substitution.
IMAGE={image}

if [ "$OFFLINE" = "1" ]; then
    if docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "Image already available: $IMAGE"
        exit 0
    fi
    echo "OFFLINE: image $IMAGE is not present on this host and offline mode does not pull" >&2
    exit 3
fi

if [ "$FORCE_PULL" != "1" ] && docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image already available: $IMAGE"
    exit 0
fi

if [ "$FORCE_PULL" = "1" ]; then
    echo "Force pull requested for image: $IMAGE..."
else
    echo "Pulling image: $IMAGE..."
fi
docker pull "$IMAGE"
