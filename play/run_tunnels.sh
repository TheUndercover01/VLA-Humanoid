#!/bin/bash
# 15 Oct: open the two games to the internet with Cloudflare quick tunnels (no account; the address changes on every start) and publish the
# addresses for the site page projects/play/ in ayushdeshmukh/projects/play/servers.json (committed + pushed with PUSH=1).
#   play/run_sabotage.sh start; play/run_playground.sh start     (the games, bound to 127.0.0.1)
#   PUSH=1 play/run_tunnels.sh start                             (tunnels + servers.json [+ push the site])
#   play/run_tunnels.sh stop                                     (tunnels down; servers.json emptied [+ pushed with PUSH=1])
cd "$(dirname "$0")/.."
C=/media/storage/ayush/cache; CF=${CF:-$HOME/bin/cloudflared}; SITE=../ayushdeshmukh; OUT=$SITE/projects/play/servers.json
declare -A PORTS=([sabotage]=8810 [playground]=8800)
publish() {
  [ -n "$PUSH" ] || return 0
  (cd $SITE && git add projects/play/servers.json && git commit -q -m "play: robot $1" && git push -q git@github.com:TheUndercover01/ayushdeshmukh.git main && echo "site pushed")
}
stop() {
  for g in "${!PORTS[@]}"; do [ -f $C/tunnel_$g.pid ] && kill $(cat $C/tunnel_$g.pid) 2>/dev/null; rm -f $C/tunnel_$g.pid; done
}
case $1 in
  stop) stop; echo "{}" > $OUT; publish offline ;;
  start)
    stop
    pairs=""
    for g in sabotage playground; do
      curl -sf http://127.0.0.1:${PORTS[$g]}/health > /dev/null || { echo "$g is not running on ${PORTS[$g]}: start it first"; continue; }
      : > $C/tunnel_$g.log
      setsid $CF tunnel --no-autoupdate --url http://127.0.0.1:${PORTS[$g]} > $C/tunnel_$g.log 2>&1 < /dev/null & echo $! > $C/tunnel_$g.pid
      for _ in $(seq 60); do url=$(grep -aoE 'https://[a-z0-9-]+\.trycloudflare\.com' $C/tunnel_$g.log | head -1); [ -n "$url" ] && break; sleep 1; done
      [ -n "$url" ] || { echo "$g: no tunnel address, see $C/tunnel_$g.log"; continue; }
      echo "$g -> $url"
      pairs="$pairs $g=$url"
    done
    python3 -c "import json,sys; print(json.dumps(dict(a.split('=',1) for a in sys.argv[1:]), indent=2))" $pairs > $OUT; cat $OUT
    publish online ;;
esac
