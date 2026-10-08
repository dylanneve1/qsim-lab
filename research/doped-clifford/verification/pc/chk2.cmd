@echo off
set TEMP=E:\qsim-verify\tmp
set TMP=E:\qsim-verify\tmp
cd /d E:\qsim-verify
E:\qsim-runprep-gpu-target\release\examples\chain_sweep.exe tailcheck --tail --d 40 --n 70 --m 6 --k 3 --backends cpu64,int5:b16:h --l 18 --slots 14 > tc40.log 2>&1
echo done >> tc40.log
