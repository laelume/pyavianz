#!/bin/bash
# List the kept recordings (and their annotations) that start between 19:00 and
# 06:59, judged from the YYYYMMDD_HHMMSS (or YYMMDD_HHMMSS) time in the filename.
# Paths are relative to the results folder, ready for `tar -T`.
#
#   bash night_list.sh [results folder] > night.txt
#
# Counts for each stage go to stderr, so a zero shows where files were lost.

dir=${1:-/nfs/scratch/$USER/solar_survey/Kiwi_Nth_Is_Brown}
cd "$dir" || exit 1

all=$(find . -name '*.data' -not -path './annotations/*' | sort)
timed=$(grep -E '[0-9]{6,8}_[0-9]{6}[^/]*\.data$' <<< "$all")
night=$(grep -E '[0-9]{6,8}_(19|2[0-3]|0[0-6])[0-9]{4}[^/]*\.data$' <<< "$timed")

echo "kept annotations:       $(grep -c . <<< "$all")" >&2
echo "  with a time in name:  $(grep -c . <<< "$timed")" >&2
echo "  starting 19:00-06:59: $(grep -c . <<< "$night")" >&2
if [ -z "$night" ]; then
    echo "sample names:" >&2
    head -5 <<< "$all" >&2
fi

while read -r f; do
    [ -n "$f" ] && echo "$f" && echo "${f%.data}"
done <<< "$night"
