#!/bin/bash
# P2 Edge0-on-our-harness: 2 routing modes × 3 fresh processes, median decode tok/s
set -u
EI="competitors/Edge0/edge0-infer/build/mac-metal/bin/edge0-infer"
export MOE_ASSETS_35B="/Users/youssefhariri/Documents/ideas_2026/New_Hanzi/moe_serving_research/models.nosync/edge0-35b-assets"
OUT="results_P2/edge0_harness.jsonl"
> "$OUT"
for mode in actual predicted; do
  for rep in 1 2 3; do
    echo "RUN $mode rep$rep"
    "$EI" bench --model "$MOE_ASSETS_35B" --device metal \
      --model-option routing=$mode \
      --user-file bench_prompt.txt --warmup 10 --ntok 100 --runs 1 \
      | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if line.startswith('{'):
        d = json.loads(line)
        d['routing_mode'] = '$mode'
        d['rep'] = $rep
        print(json.dumps(d))
" >> "$OUT"
  done
done
echo DONE
