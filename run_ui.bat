@echo off
rem Pokemon Champions battle assistant - Windows launcher. Double-click this file.
rem
rem   run_ui.bat                    start it and open the browser at http://127.0.0.1:8765
rem   run_ui.bat --host 0.0.0.0     phones on the same Wi-Fi can open it too, the address is printed
rem   run_ui.bat --port 8766        use another port
rem
rem The first run creates a Python environment in LOCALAPPDATA\pokechamp\venv, a short path outside this
rem folder, and installs numpy and PyTorch into it: about 200 MB, a few minutes. Later runs start right away,
rem also from a newly downloaded copy of this folder. Delete that venv folder to remove the packages again.
rem Optional: set POKECHAMP_VENV to use another folder for the environment.
setlocal
chcp 65001 >nul
set "HERE=%~dp0"
cd /d "%HERE%" 2>nul || pushd "%HERE%"
if not exist "pokechamp\__main__.py" goto wrong_folder

rem Where the environment lives: POKECHAMP_VENV, else a finished .venv next to this file, else LOCALAPPDATA
set "VENV=%HERE%.venv"
if defined POKECHAMP_VENV set "VENV=%POKECHAMP_VENV%"
if defined POKECHAMP_VENV goto check
if exist "%HERE%.venv\pokechamp-deps-ok" goto check
if defined LOCALAPPDATA set "VENV=%LOCALAPPDATA%\pokechamp\venv"

:check
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\pokechamp-deps-ok"
if not exist "%MARKER%" goto setup
"%VPY%" -c "import sys" <nul >nul 2>&1
if errorlevel 1 goto setup
goto start

:setup
if not exist "%VENV%" goto create
if not exist "%VENV%\pyvenv.cfg" goto not_a_venv
echo Setting up the Python environment again: "%VENV%"
rmdir /s /q "%VENV%"
if exist "%VENV%" goto venv_locked

:create
call :find_python
if not defined PY goto no_python
echo.
%PY% -c "import sys; print('Using Python', sys.version.split()[0], sys.executable)"
%PY% -c "import sys; sys.exit(0 if sys.version_info[:2] <= (3, 14) else 1)" <nul >nul 2>&1
if errorlevel 1 echo Note: PyTorch may not support this new Python version yet. If the installation fails, install Python 3.13.
echo Creating the Python environment in "%VENV%" ...
%PY% -m venv "%VENV%"
if errorlevel 1 goto venv_failed
if exist "%VPY%" goto install
rem Microsoft Store Python may hide what it writes under AppData from other programs: use this folder instead.
if /i "%VENV%"=="%HERE%.venv" goto venv_failed
set "VENV=%HERE%.venv"
goto check

:install
echo.
echo Installing numpy and PyTorch. This happens only once: about 200 MB download, a few minutes.
"%VPY%" -m pip install --disable-pip-version-check numpy torch
if errorlevel 1 goto pip_failed
"%VPY%" -c "import numpy, torch" <nul
if errorlevel 1 goto import_failed
echo ok>"%MARKER%"
echo Installation finished.

:start
echo.
"%VPY%" -m pokechamp ui --open %*
if errorlevel 1 goto server_failed
endlocal
exit /b 0

rem ---------------------------------------------------------------------------------------------------------
:find_python
rem Prefers a Python that PyTorch supports, 3.10 to 3.14, then any Python 3.10 or newer.
set "PY="
call :try_python 14 py -3
for %%V in (3.13 3.12 3.14 3.11 3.10) do if not defined PY call :try_python 14 py -%%V
if not defined PY call :try_python 14 python
if not defined PY call :try_python 99 py -3
if not defined PY call :try_python 99 python
goto :eof

:try_python
rem Arguments: the highest accepted minor version, then the command. Sets PY when that Python fits.
%2 %3 -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, %1) else 1)" <nul >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%2 %3"
goto :eof

rem ---------------------------------------------------------------------------------------------------------
:wrong_folder
echo.
echo [ERROR] The pokechamp folder was not found next to this file.
echo   Extract the whole ZIP file first: right-click it, choose "Extract All", then open the extracted
echo   folder and double-click run_ui.bat there. Do not run it from inside the ZIP window.
goto fail

:no_python
echo.
echo [ERROR] Python 3.10 or newer was not found.
echo   1. Download Python 3.13 from https://www.python.org/downloads/
echo   2. In the installer, tick the box "Add python.exe to PATH", then click "Install Now".
echo   3. Run this file again.
goto fail

:not_a_venv
echo.
echo [ERROR] "%VENV%" exists but is not a Python environment. Move or delete it, then run this file again.
goto fail

:venv_locked
echo.
echo [ERROR] Could not delete "%VENV%". Close other battle assistant windows, then run this file again.
goto fail

:venv_failed
echo.
echo [ERROR] Could not create the Python environment in "%VENV%".
echo   Install Python 3.13 from https://www.python.org/downloads/ and run this file again.
goto fail

:pip_failed
echo.
echo [ERROR] Installing numpy and PyTorch failed. See the messages above.
"%VPY%" -c "import sys, platform; print('  This Python:', sys.version.split()[0], platform.architecture()[0])"
echo   - Check the internet connection and run this file again.
echo   - PyTorch needs 64-bit Python 3.10 to 3.14. With another version, install Python 3.13 from
echo     https://www.python.org/downloads/ and run this file again.
goto fail

:import_failed
echo.
echo [ERROR] numpy or PyTorch was installed but cannot be loaded. See the messages above.
echo   - If the message mentions a DLL, install the Microsoft Visual C++ Redistributable from
echo     https://aka.ms/vs/17/release/vc_redist.x64.exe and run this file again.
goto fail

:server_failed
echo.
echo The battle assistant stopped because of the error above.
goto fail

:fail
echo.
pause
endlocal
exit /b 1
