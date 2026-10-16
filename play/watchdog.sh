#!/bin/bash
# 16 Oct: keeps the three live games up while reviewers may play (they run on this machine behind Cloudflare quick tunnels, which can drop).
# Every 60 s, per game: the game itself (127.0.0.1 /health) dead -> restart it with its launcher; its public tunnel failing twice in a row ->
# a new tunnel for that game only, its address written to ayushdeshmukh/projects/play/servers.json and the site pushed (the page reads it).
#   setsid nohup play/watchdog.sh > /dev/null 2>&1 < /dev/null &        log: cache/watchdog.log        stop: kill $(cat cache/watchdog.pid)
cd "$(dirname "$0")/.."
C=/media/storage/ayush/cache; CF=${CF:-$HOME/bin/cloudflared}; SITE=../ayushdeshmukh; OUT=$SITE/projects/play/servers.json
PERIOD=${PERIOD:-60}
declare -A PORT=([sabotage]=8810 [playground]=8800 [tictactoe]=8820)
declare -A START=([sabotage]="GPU=0 play/run_sabotage.sh start" [playground]="GPU=0 play/run_builder.sh start"
                  [tictactoe]="GPU=1 play/run_tictactoe.sh start ours_cyl_marks_ttt_tiles")
declare -A MISS
echo $$ > $C/watchdog.pid
log() { echo "$(date '+%a %H:%M:%S') $*" >> $C/watchdog.log; }
url_of() { python3 -c "import json,sys; print(json.load(open('$OUT')).get(sys.argv[1], ''))" $1 2>/dev/null; }

new_tunnel() {   # new_tunnel GAME: a fresh quick tunnel for one game, written to servers.json, site pushed
  local g=$1 url=""
  [ -f $C/tunnel_$g.pid ] && kill $(cat $C/tunnel_$g.pid) 2>/dev/null
  : > $C/tunnel_$g.log
  setsid $CF tunnel --no-autoupdate --url http://127.0.0.1:${PORT[$g]} > $C/tunnel_$g.log 2>&1 < /dev/null & echo $! > $C/tunnel_$g.pid
  for _ in $(seq 60); do url=$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' $C/tunnel_$g.log | head -1); [ -n "$url" ] && break; sleep 1; done
  [ -n "$url" ] || { log "$g: new tunnel gave no address"; return 1; }
  for _ in $(seq 30); do curl -sf -m 5 $url/health > /dev/null && break; sleep 2; done        # DNS of a new quick tunnel takes a few s
  python3 -c "
import json, sys
p, g, u = sys.argv[1:]
d = json.load(open(p)) if open(p).read().strip() else {}
d[g] = u
open(p, 'w').write(json.dumps(d, indent=2) + '\n')" $OUT $g $url
  (cd $SITE && git add projects/play/servers.json && git commit -q -m "play: new $g address" \
     && timeout 60 git pull -q --rebase --autostash git@github.com:TheUndercover01/ayushdeshmukh.git main \
     && timeout 120 git push -q git@github.com:TheUndercover01/ayushdeshmukh.git main) && log "$g: new tunnel $url, site pushed" \
     || log "$g: new tunnel $url, site push FAILED"
}

log "watchdog started (pid $$, every ${PERIOD}s)"
while :; do
  for g in sabotage playground tictactoe; do
    if ! curl -sf -m 5 http://127.0.0.1:${PORT[$g]}/health > /dev/null; then
      log "$g: game down, restarting"
      eval "${START[$g]}" >> $C/watchdog.log 2>&1
      curl -sf -m 5 http://127.0.0.1:${PORT[$g]}/health > /dev/null && log "$g: game back" || log "$g: restart FAILED"
      continue
    fi
    u=$(url_of $g)
    if [ -n "$u" ] && curl -sf -m 10 $u/health > /dev/null; then
      MISS[$g]=0
    else
      MISS[$g]=$(( ${MISS[$g]:-0} + 1 ))
      log "$g: public address not answering (${MISS[$g]}x) $u"
      [ ${MISS[$g]} -ge 2 ] && { new_tunnel $g; MISS[$g]=0; }
    fi
  done
  sleep $PERIOD
done
