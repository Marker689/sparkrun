# TensorFold external assets (not part of the sparkrun PR)

Sparkrun's review established that runtime container definitions and recipes do
not belong in the sparkrun repository.  This branch keeps them as a separate,
convenience snapshot for hardware validation; **none of this is part of PR
#332**.

Contents:

- `docker/tensorfold/` — upstream TensorFold release image (CUDA/NCCL/CUDA
  runtime base, pinned release + sha256).
- `docker/tensorfold-dsv41/` — DeepSeek-V4.1 fork images for the TP2 lane
  (`3ff3b2a`, three patches) and TP4 lane (`5940208`, four patches), including
  the vendored patches and the 1017 chat template.
- `scripts/tensorfold-dsv41-prepare.sh` — one-time DeepSeek bundle preparation
  (rank caches, kit, Engram, model metadata).
- `recipes/` — solo Qwen/Nemotron, GLM TP2, Nemotron Super VL TP2, and
  DeepSeek TP2/TP4 examples, already adapted to the single `tensorfold` runtime
  and the `tf_family: dsv41` mode.
- `.github/workflows/tensorfold*` — build workflows for the images (useful if
  these assets are moved to their own repository).

The runtime itself (the PR) expects an upstream image tag at
`ghcr.io/spark-arena/tensorfold:1.1.3` and an explicit `container:` for each
DeepSeek recipe.  Build the images on an arm64 DGX Spark:

```sh
docker/tensorfold/build.sh --load
docker/tensorfold-dsv41/build.sh --load              # TP2
docker/tensorfold-dsv41/build.sh --lane tp4 --load    # TP4
```

Then see `docs/TENSORFOLD.md` in the PR for the runtime rules and validation
checklist.  This branch is deliberately not proposed upstream.
