#!/bin/bash
# Run a job under a HARD memory ceiling and a wall clock, so a runaway kills
# itself instead of the session.
#
#   guard_run.sh <mem> <secs> <logfile> <cmd...>    e.g. 12G 3600 x.log python a.py
#
# Exit: 0 ok | 124 wall-clock timeout | 137/143 killed at the memory cap
#       (anything else is the job's own failure, not the guard)
#
# TWO GUARDS THAT LOOK RIGHT AND ARE NOT:
#   RLIMIT_AS is useless here. Imports alone reserve ~16 GB of VIRTUAL address
#   space while RSS stays under 0.4 GB, so an address-space cap either fires on
#   import or never fires. A 6 GB limit once killed all 60 jobs instantly.
#   MemoryHigh throttles, it does not kill. With it set a runaway crawls at the
#   ceiling until the wall clock fires, wasting the timeout and then blaming
#   the clock. MemoryMax alone OOM-kills, in well under a second.
#
# TIMEOUT STAYS OUTSIDE THE SCOPE. Inside, the cgroup OOM kill takes out every
# process in the scope -- `timeout` included -- and the exit code comes back
# 124, blaming the clock for a memory kill. Outside, the causes stay apart.
set -uo pipefail
MEM=${1:?mem e.g. 12G}; SECS=${2:?seconds}; LOG=${3:?logfile}; shift 3
[ $# -ge 1 ] || { echo "guard_run: no command given" >&2; exit 2; }
mkdir -p "$(dirname "$LOG")"
{ echo "### guard_run $(date -Is)"; echo "### mem=$MEM wall=${SECS}s"
  echo "### cmd: $*"; } > "$LOG"
timeout --signal=TERM --kill-after=20 "$SECS" \
    systemd-run --user --scope -q \
    -p MemoryMax="$MEM" -p MemorySwapMax=0 \
    "$@" >>"$LOG" 2>&1
rc=$?
case $rc in
  0)        echo "guard_run: OK" ;;
  124)      echo "guard_run: TIMEOUT after ${SECS}s -- job killed, session safe" ;;
  137|143)  echo "guard_run: KILLED (memory cap $MEM) -- session safe" ;;
  *)        echo "guard_run: job exited $rc (its own failure, not the guard)" ;;
esac
echo "guard_run: log -> $LOG  (rc=$rc)"
exit $rc
