#!/bin/bash
rm -rf waiting.log || true
echo "Trying to allocate gpu"
start=$(date +%s)
until gpu-allocate-cli allocate --duration 72h --wait && echo "Elapsed time: $(( $( date +%s ) - $start )) seconds" >| waiting.log && dvc repro
do 
  echo "Waiting 30 mins"
  sleep 1800
  echo "Trying to allocate gpu"
  echo "Elapsed time: $(( $( date +%s ) - $start )) seconds"
  echo "Elapsed time in hours: $(( ($(date +%s) - $start) / 3600 )) hours"
  echo "Elapsed time in days: $(( ($(date +%s) - $start) / 86400 )) days"
done
