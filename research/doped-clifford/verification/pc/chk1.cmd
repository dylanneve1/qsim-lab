@echo off
set TEMP=E:\qsim-verify\tmp
set TMP=E:\qsim-verify\tmp
cd /d E:\qsim-verify
echo start %TIME% > chk1.log
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 48 --format cpu64 --m 6 --count 16 --seed-file seed.txt --out ref_d48.jsonl --heartbeat hb_ref.txt >> chk1.log 2>&1
echo ref_done %TIME% >> chk1.log
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --gpu --tail --d 48 --format int5:b16:h --l 18 --slots 14 --m 6 --count 16 --seed-file seed.txt --out gpu5_d48.jsonl --heartbeat hb_gpu.txt >> chk1.log 2>&1
echo gpu_done %TIME% >> chk1.log
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe run --tail --d 48 --format int5:b16:h --l 18 --slots 14 --m 6 --count 16 --seed-file seed.txt --out cpu5_d48.jsonl --heartbeat hb_cpu.txt >> chk1.log 2>&1
echo cpu_done %TIME% >> chk1.log
