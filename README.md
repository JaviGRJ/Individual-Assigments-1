# Individual-Assigments-1
Development of assigment 1 BD

$vcvars = Get-ChildItem `
  "C:\Program Files\Microsoft Visual Studio" `
  -Filter vcvars64.bat `
  -Recurse `
  -ErrorAction SilentlyContinue |
  Select-Object -First 1 -ExpandProperty FullName

cmd.exe /d /c "call `"$vcvars`" && cd /d `"C:\Users\Javier Guerra\Individual-Assigments-1`" && cl /nologo /Od /W4 /std:c11 matrix_multiply.c"

No olvidarse de que hay que ejecutar esto antes
