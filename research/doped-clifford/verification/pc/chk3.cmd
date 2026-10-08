@echo off
set TEMP=E:\qsim-verify\tmp
set TMP=E:\qsim-verify\tmp
cd /d E:\qsim-verify
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 40 --format cpu64 --m 6 --count 8 --seed-file seed.txt --out ref_d40.jsonl --heartbeat hb3.txt > chk3.log 2>&1
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --gpu --tail --d 40 --format int5:b16:h --l 18 --slots 14 --m 6 --count 8 --seed-file seed.txt --out gpu5_d40.jsonl --heartbeat hb3.txt >> chk3.log 2>&1
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 40 --format int5:b16:h --l 18 --slots 14 --m 6 --count 8 --seed-file seed.txt --out cpu5_d40.jsonl --heartbeat hb3.txt >> chk3.log 2>&1
echo done >> chk3.log
