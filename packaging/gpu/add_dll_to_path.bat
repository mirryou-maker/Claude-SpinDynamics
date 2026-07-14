@echo off
REM ===========================================================================
REM Claude-SpinDynamics - add the bundled CUDA/FFTW runtime DLLs to PATH so the
REM GPU apps (any variant's bin\*_gpu.exe) and the Python module can find them.
REM
REM   add_dll_to_path.bat            interactive menu (double-click friendly)
REM   add_dll_to_path.bat /session   add to PATH for THIS terminal only
REM   add_dll_to_path.bat /user      permanently add to the current user's PATH
REM   add_dll_to_path.bat /system    permanently add to the system PATH (admin)
REM   add_dll_to_path.bat /remove    remove it again from user & system PATH
REM ===========================================================================
setlocal EnableExtensions
set "DLLDIR=%~dp0runtime-dll"
if "%DLLDIR:~-1%"=="\" set "DLLDIR=%DLLDIR:~0,-1%"
set "SCOPE="

REM ---- command-line dispatch (an argument was given) -------------------------
if /I "%~1"=="/session"   goto :session
if /I "%~1"=="/user"      set "SCOPE=User"
if /I "%~1"=="/permanent" set "SCOPE=User"
if /I "%~1"=="/system"    set "SCOPE=Machine"
if defined SCOPE          goto :perm
if /I "%~1"=="/remove"    goto :remove
if /I "%~1"=="/help"      goto :help
if /I "%~1"=="-h"         goto :help
if /I "%~1"=="/?"         goto :help
if not "%~1"==""          goto :help

REM ---- no argument: interactive menu (double-click case) ---------------------
:menu
cls
echo ===========================================================================
echo   Claude-SpinDynamics - runtime-dll PATH setup
echo ===========================================================================
echo   Target folder:
echo     %DLLDIR%
echo.
echo   1. session  - add to PATH for THIS terminal only (does NOT persist)
echo   2. user     - permanently add to current user's PATH (no admin needed)
echo   3. system   - permanently add to system PATH for all users (admin)
echo   4. remove   - remove runtime-dll from user ^& system PATH
echo   5. exit     - do nothing and close
echo.
set "CHOICE="
set /p "CHOICE=Enter a number or name (session/user/system/remove/exit): "

if /I "%CHOICE%"=="1"       goto :session
if /I "%CHOICE%"=="session" goto :session
if /I "%CHOICE%"=="default" goto :session
if /I "%CHOICE%"=="2"       ( set "SCOPE=User"    & goto :perm )
if /I "%CHOICE%"=="user"    ( set "SCOPE=User"    & goto :perm )
if /I "%CHOICE%"=="3"       ( set "SCOPE=Machine" & goto :perm )
if /I "%CHOICE%"=="system"  ( set "SCOPE=Machine" & goto :perm )
if /I "%CHOICE%"=="4"       goto :remove
if /I "%CHOICE%"=="remove"  goto :remove
if /I "%CHOICE%"=="5"       goto :end
if /I "%CHOICE%"=="exit"    goto :end

echo.
echo Invalid input "%CHOICE%". Please try again.
echo.
pause
goto :menu

REM ---- session only ----------------------------------------------------------
:session
set "PATH=%DLLDIR%;%PATH%"
echo.
echo [session] runtime-dll added to PATH for THIS terminal only.
echo Run e.g.:  cuFFT-f64\bin\sp4_gpu.exe
echo.
echo NOTE: this change disappears when the window closes.
echo       Use option 2 (user) to make it permanent.
goto :end

REM ---- permanent add (SCOPE is User or Machine) ------------------------------
:perm
if /I not "%SCOPE%"=="Machine" goto :perm_do
REM /system needs elevation - if not admin, relaunch with a UAC prompt
net session >nul 2>&1
if not errorlevel 1 goto :perm_do
echo.
echo [system] Administrator rights are required. Opening an elevated window...
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -ArgumentList '/system' -Verb RunAs" 2>nul
if errorlevel 1 (
    echo Elevation was cancelled. Nothing changed.
    echo Tip: right-click this .bat -^> "Run as administrator", or use option 2 (user).
)
goto :end

:perm_do
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
echo Usage: add_dll_to_path.bat [ /session ^| /user ^| /system ^| /remove ]
echo.
echo   (no argument)  show interactive menu (double-click friendly)
echo   /session       add runtime-dll to PATH for the current terminal only
echo   /user          permanently add to the current user's PATH (no admin needed)
echo   /system        permanently add to the system PATH for all users (Administrator)
echo   /remove        remove runtime-dll from the user and system PATH
goto :end

:end
echo.
pause
endlocal
