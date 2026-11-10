# tensorfold-dsv41-zig image

This is the external image for the **Jayleaton Zig** DeepSeek-V4.1 engine
selected by `tf_family: jayleaton` in the sparkrun TensorFold runtime.

It is built from [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)
(`config/tensorfold.rev` / `config/tensorfold.repo`, `docker/zig/Dockerfile`).  The hardware-validated pin is
`7b79ef92d846138d51638b677c473392a56e240c` (`tensorfold-native 1.1.8`), whose rank-1 entry point is
`tf-dsv41 follow PACK ASSETS` (the older pin called it `tf-dsv41-m1`).  The image is not part of the
sparkrun package; build it on an ARM64 DGX Spark and load the same image on
both nodes:

```bash
git clone https://github.com/jayleaton/deepseek-v41-tensorfold-spark.git
cd deepseek-v41-tensorfold-spark
mkdir -p build/zig/src build/zig/fatbins
cp -a engine/zig/. build/zig/src/
cp docker/zig/Dockerfile build/zig/Dockerfile
touch build/zig/fatbins/.keep
# The stock Dockerfile downloads Zig from ziglang.org.  On a slow link, place an
# unpacked 0.17.0 compiler in build/zig/zig, replace the ZIG_URL RUN block with
# `COPY zig /opt/zig` + `RUN /opt/zig/zig version`, and build:
cd build/zig
docker build -t tensorfold-dsv41-zig:local \
  --build-arg ZIG_COMMIT="$(git -C ../../ rev-parse HEAD)" \
  --build-arg CUDA_DEVEL=nvidia/cuda:13.0.1-devel-ubuntu24.04 \
  --build-arg CUDA_RUNTIME=nvidia/cuda:13.0.1-base-ubuntu24.04 .
```

The runtime assets (per-rank Engram bins, `rope.bin`, `engram-token-map.bin`, AOT
cubins) are generated with the same upstream repository's install notes and are
not baked into this image.  The current pin opens `engram-token-map.bin`; a
legacy `engram-host.bin` is not enough.  Their host paths are mounted by the recipe in
`recipes/jayleaton-dsv41-flash-tp2-ablit.yaml`.

The model pack is the jayleaton flash17 2.8bpw pack with the drowzeys Keys
L10-35 `wo_b` overlay applied; the recipe references it through
`cluster_config.resolved_model_path` and the synthetic HF-cache id
`marker/DeepSeek-V4.1-Flash-1017-EXL3-2.8bpw-Abliterated`.
