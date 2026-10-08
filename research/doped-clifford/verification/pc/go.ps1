param([string]$f)
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine="cmd /c start `"`" /belownormal /b /wait cmd /c E:\qsim-verify\$f"; CurrentDirectory='E:\qsim-verify'}
"pid=$($r.ProcessId) rc=$($r.ReturnValue)"
