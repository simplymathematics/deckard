#!/bin/bash
rm -rf waiting.log || true
echo "Trying to allocate gpu"
start=$(date +%s)
echo "Start time (seconds): $start" >| waiting.log
until gpu-allocate-cli allocate --duration 72h --wait && echo "Elapsed time: $(( $( date +%s ) - $start )) seconds" >| waiting.log #&& dvc repro
do 
  echo "Trying to allocate gpu"
  echo "Elapsed time: $(( $( date +%s ) - $start )) seconds"
  echo "Elapsed time: $(( $( date +%s ) - $start )) seconds" >> waiting.log
  echo "Elapsed time in hours: $(( ($(date +%s) - $start) / 3600 )) hours"
  echo "Elapsed time in hours: $(( ($(date +%s) - $start) / 3600 )) hours" >> waiting.log
  echo "Elapsed time in days: $(( ($(date +%s) - $start) / 86400 )) days"
  echo "Elapsed time in days: $(( ($(date +%s) - $start) / 86400 )) days" >> waiting.log
  echo "Waiting 10 minutes before trying to allocate gpu again"
  sleep 10m
done