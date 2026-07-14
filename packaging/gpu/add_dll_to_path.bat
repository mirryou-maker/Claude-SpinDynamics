@echo off
REM ===========================================================================
REM Claude-SpinDynamics - add the bundled CUDA/FFTW runtime DLLs to PATH so the
REM GPU apps (any variant's bin\*_gpu.exe) and the Python module can find them.
REM
REM   add_dll_to_path.bat            add to PATH for THIS terminal only (default)
REM   add_dll_to_path.bat /user      permanently add to the current user's PATH
REM   add_dll_to_path.bat /system    permanently add to the system PATH (admin)
REM   add_dll_to_path.bat /remove    remove it again from user & system PATH
REM ===========================================================================
setlocal
set "DLLDIR=%~dp0runtime-dll"
if "%DLLDIR:~-1%"=="\" set "DLLDIR=%DLLDIR:~0,-1%"

if /I "%~1"=="/user"      goto :perm
if /I "%~1"=="/permanent" goto :perm
if /I "%~1"=="/system"    goto :perm
if /I "%~1"=="/remove"    goto :remove
if /I "%~1"=="/help"      goto :help
if /I "%~1"=="-h"         goto :help
if /I "%~1"=="/?"         goto :help
if not "%~1"==""          goto :help

REM ---- default: current terminal session only --------------------------------
set "PATH=%DLLDIR%;%PATH%"
echo [session] runtime-dll added to PATH for THIS terminal only.
echo Run e.g.:  cuFFT-f64\bin\sp4_gpu.exe
echo.
echo To make it PERMANENT (survives reboot, applies to new terminals):
echo   add_dll_to_path.bat /user     add to your account's PATH ^(no admin^)
echo   add_dll_to_path.bat /system   add to the system PATH for all users ^(run as Administrator^)
echo   add_dll_to_path.bat /remove   undo ^(remove from user ^& system PATH^)
goto :end

REM ---- permanent add (User by default, Machine for /system) -------------------
:perm
set "SCOPE=User"
if /I "%~1"=="/system" set "SCOPE=Machine"
if /I "%SCOPE%"=="Machine" (
    net session >nul 2>&1
    if errorlevel 1 (
        echo ERROR: /system requires an Administrator command prompt.
        echo Right-click cmd.exe -^> "Run as administrator", then re-run this.
        goto :end
    )
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d='%DLLDIR%'; $s='%SCOPE%'; $p=[Environment]::GetEnvironmentVariable('Path',$s); $items= if($p){$p.Split(';')|Where-Object{$_ -ne ''}}else{@()}; if($items -contains $d){Write-Host ('['+$s+'] already in PATH: '+$d)}else{$np= if($p){$p.TrimEnd(';')+';'+$d}else{$d}; [Environment]::SetEnvironmentVariable('Path',$np,$s); Write-Host ('['+$s+'] permanently added to PATH: '+$d)}"
echo.
echo Done. Open a NEW terminal (or log off/on) for the change to take effect.
goto :end

REM ---- remove from both user and system PATH ---------------------------------
:remove
powershell -NoProfile -ExecutionPolicy Bypass -Command "$d='%DLLDIR%'; foreach($s in 'User','Machine'){ try{$p=[Environment]::GetEnvironmentVariable('Path',$s)}catch{continue}; if(-not $p){continue}; $np=($p.Split(';')|Where-Object{$_ -ne '' -and $_ -ne $d}) -join ';'; if($np -ne $p){ try{[Environment]::SetEnvironmentVariable('Path',$np,$s); Write-Host ('['+$s+'] removed: '+$d)}catch{Write-Host ('['+$s+'] found but needs admin to remove')} } else { Write-Host ('['+$s+'] not present') } }"
echo.
echo Done. Open a NEW terminal for the change to take effect.
goto :end

:help
echo Usage: add_dll_to_path.bat [ /user ^| /system ^| /remove ]
echo.
echo   (no argument)  add runtime-dll to PATH for the current terminal only
echo   /user          permanently add to the current user's PATH (no admin needed)
echo   /system        permanently add to the system PATH for all users (Administrator)
echo   /remove        remove runtime-dll from the user and system PATH
goto :end

:end
endlocal
