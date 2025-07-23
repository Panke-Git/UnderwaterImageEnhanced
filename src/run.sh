#!/bin/bash

#############################################
# 自动训练启动脚本 - 运行多个模型训练任务
# 每个任务独立 tmux 会话 + 自动日志记录
#############################################

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pax

PROJECT_ROOT="/root/cyx/CL/PRO/MyPro/UnderwaterImageEnhanced"
export PYTHONPATH="$PROJECT_ROOT"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

 训练脚本和日志路径
tmux new-session -d -s train "cd $PROJECT_ROOT && python src/train.py 2>&1 | tee $LOG_DIR/train.log"
tmux new-session -d -s train2 "cd $PROJECT_ROOT && python src/train2.py 2>&1 | tee $LOG_DIR/train2.log"
tmux new-session -d -s train3 "cd $PROJECT_ROOT && python src/train3.py 2>&1 | tee $LOG_DIR/train3.log"
tmux new-session -d -s train4 "cd $PROJECT_ROOT && python src/train4.py 2>&1 | tee $LOG_DIR/train4.log"

echo "所有训练任务已启动。"
echo "当前 tmux 会话："
tmux ls
echo "用命令 'tmux attach -t <session_name>' 进入对应会话查看输出。"