#!/bin/bash
# Start / stop the playground: (the student's policy server +) the Isaac playground on one GPU, PIDs in cache/play_*.pid
# (killing by pattern can hit the calling shell).
#   play/run_playground.sh start [checkpoint name]            the student (default ours_cyl_marks_realref until the game student exists)
#   BACKEND=teacher TEACHER=runs/rl/game_mix_s2/model_N.pt play/run_playground.sh start     the privileged teacher as the robot (tests)
#   play/run_playground.sh stop
# GPU=1 PORT=8800 SLOTS=4. The page and the robot share the port: http://localhost:$PORT/ and /site/projects/robot-playground/
cd "$(dirname "$0")/.."
export VLA_OBJECTS=multi VLA_GAME=mix VLA_TRAIN_TYPES="0,1,3,4" VLA_HOLDOUT=1 VLA_FIXED_TARGET="0.03,-0.12" VLA_PUSH_C3=0.5
C=/media/storage/ayush/cache; D=/media/storage/ayush/vla_data
GPU=${GPU:-1}; PORT=${PORT:-8800}; VPORT=${VPORT:-6201}; SLOTS=${SLOTS:-4}; NAME=${2:-ours_cyl_marks_realref}; BACKEND=${BACKEND:-vla}
stop() {
  for f in $C/play_sim.pid $C/play_server.pid; do
    [ -f $f ] && kill -9 $(cat $f) 2>/dev/null; rm -f $f
  done
  for p in $(ps -eo pid,cmd | awk '/[i]saaclab\/bin\/python play\/playground.py/ {print $1}'); do kill -9 $p; done
}
case $1 in
  stop) stop ;;
  start)
    stop
    python3 play/make_pages.py > /dev/null
    if [ $BACKEND = vla ]; then
      (export HF_HOME=$C/hf CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$PWD
       setsid /media/storage/ayush/miniconda3/envs/lerobot/bin/python -m vla.server --policy $D/train/$NAME/checkpoints/last/pretrained_model \
         --port $VPORT > $C/play_server.log 2>&1 < /dev/null & echo $! > $C/play_server.pid)
      until grep -q serving $C/play_server.log 2>/dev/null; do sleep 3; done
      EXTRA=""
    else
      EXTRA="--backend teacher --teacher $TEACHER"
    fi
    : > $C/playground.log
    CUDA_VISIBLE_DEVICES=$GPU setsid scripts/run_isaac.sh $C/playground.log play/playground.py --headless --device cuda:0 \
      --vla_port $VPORT --port $PORT --slots $SLOTS --site ../ayushdeshmukh $EXTRA > /dev/null 2>&1 < /dev/null & echo $! > $C/play_sim.pid
    until grep -aq "playground on\|Traceback" $C/playground.log; do sleep 3; done
    grep -a "playground on\|Traceback" -A8 $C/playground.log | grep -av Warning | head -12
    echo "robot: $([ $BACKEND = vla ] && echo student $NAME || echo teacher $TEACHER)" ;;
esac
