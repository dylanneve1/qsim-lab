#!/bin/bash
# runs queue.txt lines "ENV... name args..." sequentially, one locked job each
cd ~/qsim-noise-data
while read -r line; do
  [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
  name=$(echo $line | awk '{for(i=1;i<=NF;i++) if($i !~ /=/){print $i; exit}}')
  [ -f "$name.csv" ] && continue
  env $(echo $line | tr ' ' '\n' | grep '=' | tr '\n' ' ') ./job.sh $(echo $line | tr ' ' '\n' | grep -v '=' | tr '\n' ' ') >> queue.log 2>&1
done < queue.txt
echo QUEUE-DONE >> queue.log
