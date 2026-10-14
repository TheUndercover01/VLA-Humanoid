#!/bin/bash
# Start / stop the sentence builder in the robot's own scene (play/builder.py): the student's policy server + the Isaac game on one GPU, PIDs in cache/bld_*.pid.
#   play/run_builder.sh start [checkpoint name]     default ours_cyl_marks_nolift (the final model, 14 Oct)
#   play/run_builder.sh stop
# GPU=1 PORT=8800 VPORT=6201 SLOTS=4. Page: http://127.0.0.1:$PORT/ (binds 127.0.0.1 only)
cd "$(dirname "$0")/.."
export VLA_OBJECTS=cube_cylinder VLA_PUSH_C3=0.5 VLA_FIXED_TARGET="0.03,-0.12"
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data
GPU=${GPU:-1}; PORT=${PORT:-8800}; VPORT=${VPORT:-6201}; SLOTS=${SLOTS:-4}; NAME=${2:-ours_cyl_marks_nolift}
stop() {
  for f in $C/bld_sim.pid $C/bld_server.pid; do
    [ -f $f ] && kill -9 $(cat $f) 2>/dev/null; rm -f $f
  done
  for p in $(ps -eo pid,cmd | awk '/[i]saaclab\/bin\/python play\/builder.py/ {print $1}'); do kill -9 $p; done
}
case $1 in
  stop) stop ;;
  start)
    stop
    (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
     setsid /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model \
       --port $VPORT > $C/bld_server.log 2>&1 < /dev/null & echo $! > $C/bld_server.pid)
    until grep -q serving $C/bld_server.log 2>/dev/null; do sleep 3; done
    : > $C/builder.log
    CUDA_VISIBLE_DEVICES=$GPU setsid scripts/run_isaac.sh $C/builder.log play/builder.py --headless --device cuda:0 \
      --vla_port $VPORT --port $PORT --slots $SLOTS > /dev/null 2>&1 < /dev/null & echo $! > $C/bld_sim.pid
    until grep -aq "builder on\|Traceback" $C/builder.log; do sleep 3; done
    grep -a "builder on\|Traceback" -A8 $C/builder.log | grep -av Warning | head -12
    echo "robot: student $NAME" ;;
esac
