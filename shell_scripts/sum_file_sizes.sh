#!/usr/bin/env bash

# this script sums up the file sizes output by calling "ls -la" on the given input
FILE_SIZES=$(ls -la $1 | tr -s ' ' | cut -f 5 -d ' ')
echo $FILE_SIZES | tr ' ' '\n' | awk '{s+=$1} END {print s}'
