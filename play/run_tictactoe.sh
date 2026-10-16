#!/bin/bash
# Start / stop the tic-tac-toe game (play/tictactoe.py): the student's policy server + the Isaac game on one GPU, PIDs in cache/ttt_*.pid.
#   play/run_tictactoe.sh start [checkpoint name]     default ours_cyl_marks_ttt (fine-tuned on the 9 board squares)
#   play/run_tictactoe.sh stop
# GPU=1 PORT=8820 VPORT=6203 SLOTS=2. Page: http://127.0.0.1:$PORT/ (binds 127.0.0.1 only)
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET=board
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data
GPU=${GPU:-1}; PORT=${PORT:-8820}; VPORT=${VPORT:-6203}; SLOTS=${SLOTS:-2}; NAME=${2:-ours_cyl_marks_ttt}
stop() {
  for f in $C/ttt_sim.pid $C/ttt_server.pid; do
    [ -f $f ] && kill -9 $(cat $f) 2>/dev/null; rm -f $f
  done
  for p in $(ps -eo pid,cmd | awk '/[i]saaclab\/bin\/python play\/tictactoe.py/ {print $1}'); do kill -9 $p; done
}
case $1 in
  stop) stop ;;
  start)
    stop
    (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
     setsid /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model \
       --port $VPORT > $C/ttt_server.log 2>&1 < /dev/null & echo $! > $C/ttt_server.pid)
    until grep -q serving $C/ttt_server.log 2>/dev/null; do sleep 3; done
    : > $C/tictactoe.log
    CUDA_VISIBLE_DEVICES=$GPU setsid scripts/run_isaac.sh $C/tictactoe.log play/tictactoe.py --headless --device cuda:0 \
      --vla_port $VPORT --port $PORT --slots $SLOTS > /dev/null 2>&1 < /dev/null & echo $! > $C/ttt_sim.pid
    until grep -aq "tictactoe on\|Traceback" $C/tictactoe.log; do sleep 3; done
    grep -a "tictactoe on\|Traceback" -A8 $C/tictactoe.log | grep -av Warning | head -12
    echo "robot: student $NAME" ;;
esac
