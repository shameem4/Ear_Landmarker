#!/bin/bash
# Thin wrapper. The implementation -- and the reasoning behind each guard
# (RLIMIT_AS is useless, MemoryHigh throttles instead of killing, timeout stays
# outside the scope, <mem> must carry a unit) -- lives in ONE place:
#
#   /mnt/14BE47C2BE479ADE/Code/utilities/guard_run.sh
#
# Kept as a wrapper because three divergent copies is how a fix lands in one
# project and leaves the others broken: on 2026-10-16 a unitless mem= made a
# 28-byte cgroup, and only this project's copy was patched.
#
# the implementation lives in the utilities git repo (Code/utilities)
# so it has history; the check below only catches an unmounted drive.
set -u
IMPL=/mnt/14BE47C2BE479ADE/Code/utilities/guard_run.sh
if [ ! -x "$IMPL" ]; then
    echo "guard_run: implementation missing or not executable:" >&2
    echo "  $IMPL" >&2
    echo "guard_run: wrap_work is not version-controlled -- restore it before running heavy jobs." >&2
    exit 127
fi
exec "$IMPL" "$@"
