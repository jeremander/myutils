#!/usr/bin/env bash

# this script sums newline-separated numbers
awk '{s+=$1} END {print s}'
