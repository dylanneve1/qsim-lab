@echo off
set TEMP=E:\qsim-verify\tmp
set TMP=E:\qsim-verify\tmp
set CS_AVX512=0
set CS_DENSE=0
set CS_BLOCK_BYTES=32768
cd /d E:\qsim-verify
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 40 --format int5:b16:h --l 18 --slots 14 --m 6 --count 8 --seed-file seed.txt --out cpuM_d40.jsonl --heartbeat hb4.txt > chk4.log 2>&1
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 48 --format int5:b16:h --l 18 --slots 14 --m 6 --count 16 --seed-file seed.txt --out cpuM_d48.jsonl --heartbeat hb4.txt >> chk4.log 2>&1
echo done >> chk4.log
