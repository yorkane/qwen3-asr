#!/bin/bash
#
# 镜像瘦身：把已构建的镜像压平成单层并删除冗余文件。
# 原理：docker export 导出容器文件系统后重新 import，删除项真正从磁盘消失
# （普通新增 RUN 层删除只是 whiteout，不减小体积）。
# 不触碰基础镜像的构建层，原有构建缓存全部可复用。
#
# 用法: ./scripts/slim-image.sh [源镜像] [目标镜像]
#   默认: w217/qwen3-asr:async-baked -> w217/qwen3-asr:async-baked-slim

set -euo pipefail

SRC="${1:-w217/qwen3-asr:async-baked}"
DST="${2:-w217/qwen3-asr:async-baked-slim}"
TMP_CONTAINER="asr-slim-tmp-$$"

info() { echo "[slim] $1"; }

size_of() { docker image inspect "$1" --format '{{.Size}}' 2>/dev/null | awk '{printf "%.1fGB", $1/1024/1024/1024}'; }

info "源镜像: $SRC ($(size_of $SRC))"

# 1) 创建临时容器（不启动完整服务），执行清理
docker create --name "$TMP_CONTAINER" --entrypoint /bin/sleep "$SRC" infinity >/dev/null
docker start "$TMP_CONTAINER" >/dev/null

info "清理冗余文件..."
docker exec "$TMP_CONTAINER" /bin/bash -c '
  set -e
  du_before=$(du -s --block-size=1M / | cut -f1)

  # --- 构建期缓存（运行时零价值）---
  rm -rf /root/.cache/uv /root/.cache/pip 2>/dev/null || true

  # --- 冗余的系统级 Python 栈（应用只用 /opt/venv，dist-packages 不在 sys.path）---
  rm -rf /usr/local/lib/python3.12/dist-packages

  # --- flashinfer Blackwell(sm100/sm103) cubins：L40/L20 等 sm_89 卡用不到 ---
  find /opt/venv/lib/python3.12/site-packages/flashinfer_cubin/cubins \
      \( -name "*sm100*" -o -name "*Sm100*" -o -name "*sm103*" -o -name "*Sm103*" \) \
      -type f -delete 2>/dev/null || true

  # --- pycache（保留包内 tests/：numpy/scipy/modelscope 有间接导入依赖）---
  find /opt/venv /app -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  find /opt/venv /app -name "*.pyc" -delete 2>/dev/null || true

  du_after=$(du -s --block-size=1M / | cut -f1)
  echo "[slim] 清理: ${du_before}MB -> ${du_after}MB (释放 $((du_before-du_after))MB)"
'

docker stop "$TMP_CONTAINER" >/dev/null

# 2) 导出并重新导入（单层、真正减小）
info "导出容器文件系统..."

# Inherit the source image's ENV (so CUDA_HOME / LD_LIBRARY_PATH /
# VLLM_USE_FLASHINFER_SAMPLER etc. survive the squash), then layer the
# minimal entrypoint/cmd/working-dir overrides on top.
ENV_CHANGES=()
while IFS='=' read -r k v; do
  [[ -z "$k" ]] && continue
  ENV_CHANGES+=(--change "ENV ${k}=${v}")
done < <(docker image inspect "$SRC" --format '{{range .Config.Env}}{{println .}}{{end}}')

docker export "$TMP_CONTAINER" | docker import \
  "${ENV_CHANGES[@]}" \
  --change 'WORKDIR /app' \
  --change 'EXPOSE 8000' \
  --change 'ENTRYPOINT ["/app/scripts/docker/entrypoint.sh"]' \
  --change 'CMD ["/opt/venv/bin/python", "start.py"]' \
  - "$DST"

docker rm "$TMP_CONTAINER" >/dev/null

info "完成: $DST ($(size_of $DST))  <-  $SRC ($(size_of $SRC))"
