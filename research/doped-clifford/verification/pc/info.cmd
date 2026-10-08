@echo off
set TEMP=E:\qsim-verify\tmp
set TMP=E:\qsim-verify\tmp
cd /d E:\qsim-verify
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe tailinfo --tail --d 48 --m 6 --format int5:b16:h --l 18 --slots 14 > info48.log 2>&1
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe tailinfo --tail --d 48 --m 6 --format int5:b16:h --gpu --l 18 --slots 14 >> info48.log 2>&1
