#!/bin/bash
# 证件照生成器 - 一键启动脚本

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================="
echo "  证件照智能生成器"
echo "==================================="

# 检查 Python
if ! command -v python3 &>/dev/null; then
    echo "未找到 python3，请先安装 Python 3.9+"
    exit 1
fi

# 创建虚拟环境（如不存在）
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv "$SCRIPT_DIR/.venv"
fi

PYTHON="$SCRIPT_DIR/.venv/bin/python3"
PIP="$SCRIPT_DIR/.venv/bin/pip"
STREAMLIT="$SCRIPT_DIR/.venv/bin/streamlit"

echo "升级 pip..."
"$PYTHON" -m pip install --upgrade pip -q

echo "安装依赖..."
"$PIP" install -q streamlit rembg Pillow numpy opencv-python-headless onnxruntime

echo ""
echo "启动应用..."
echo "地址: http://localhost:8501"
echo "按 Ctrl+C 退出"
echo ""

"$STREAMLIT" run app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
