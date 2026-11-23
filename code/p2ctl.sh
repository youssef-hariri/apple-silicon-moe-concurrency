#!/bin/bash
# p2ctl — Phase 2 experiment control (pause/resume/status/stop)
# Usage:  ./p2ctl.sh pause | resume | status | stop
DIR="$(cd "$(dirname "$0")" && pwd)"
PAUSE_FILE="$DIR/PAUSE"
STOP_FILE="$DIR/STOP"
QUEUE_DONE="$DIR/results_P2/phase2"

case "$1" in
  pause)
    touch "$PAUSE_FILE"
    echo "⏸  Pause requested. The current run unit will finish (a few minutes"
    echo "   at most), checkpoint, and the runner will wait. Nothing is lost."
    ;;
  resume)
    rm -f "$PAUSE_FILE" "$STOP_FILE"
    echo "▶  Resumed. Tell the agent (or re-run p2_runner.py) to continue the queue."
    ;;
  stop)
    touch "$STOP_FILE"
    echo "⏹  Stop requested. Runner exits after the current unit checkpoints."
    ;;
  status)
    if [ -f "$STOP_FILE" ]; then echo "state: STOP requested"; fi
    if [ -f "$PAUSE_FILE" ]; then echo "state: PAUSED (after current unit)"; else echo "state: free to run"; fi
    n=$(find "$QUEUE_DONE" -name "*.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "completed run units: $n"
    echo "latest:"
    find "$QUEUE_DONE" -name "*.json" -exec ls -t {} + 2>/dev/null | head -3 | sed 's|.*/|  |'
    ;;
  *)
    echo "Usage: $0 pause | resume | status | stop"; exit 1 ;;
esac
