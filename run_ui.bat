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
rem When PyTorch cannot be installed for this Python or computer, the assistant starts with the weaker
rem heuristic AI instead, and the next start tries PyTorch again.
rem Optional: set POKECHAMP_VENV to use another folder for the environment. A folder that this file did not
rem create is never deleted: the packages are installed into it.
setlocal
chcp 65001 >nul
set "HERE=%~dp0"
cd /d "%HERE%" 2>nul || pushd "%HERE%"
if not exist "pokechamp\__main__.py" goto wrong_folder

rem Where the environment lives: POKECHAMP_VENV, else a set-up .venv next to this file, else LOCALAPPDATA
set "VENV=%HERE%.venv"
set "OWN_DEFAULT="
if defined POKECHAMP_VENV set "VENV=%POKECHAMP_VENV%"
if defined POKECHAMP_VENV if "%VENV:~-1%"=="\" set "VENV=%VENV:~0,-1%"
if defined POKECHAMP_VENV goto check
if exist "%HERE%.venv\pokechamp-deps-ok" goto check
if exist "%HERE%.venv\pokechamp-numpy-ok" goto check
if not defined LOCALAPPDATA goto check
set "VENV=%LOCALAPPDATA%\pokechamp\venv"
rem Only this file uses that folder, an older run_ui.bat too: it counts as created by this file.
set "OWN_DEFAULT=1"

:check
rem Marker files in the environment: deps-ok = numpy and PyTorch work, numpy-ok = numpy works,
rem pokechamp-venv = this file created the environment, so only then may it delete it.
set "VPY=%VENV%\Scripts\python.exe"
set "MARKER=%VENV%\pokechamp-deps-ok"
set "NUMPY_OK=%VENV%\pokechamp-numpy-ok"
set "TAG=%VENV%\pokechamp-venv"
set "NO_MODEL="
set "PIP_QUICK="
if not exist "%MARKER%" goto setup
"%VPY%" -c "import sys" <nul >nul 2>&1
if errorlevel 1 goto setup
if not errorlevel 0 goto setup
goto start

:setup
if not exist "%VENV%" goto new_venv
if defined OWN_DEFAULT if not exist "%TAG%" echo created by run_ui.bat>"%TAG%"
if exist "%TAG%" if not exist "%VENV%\pyvenv.cfg" goto renew_venv
if not exist "%VENV%\pyvenv.cfg" goto not_a_venv
if not exist "%TAG%" goto existing_venv
rem Created by this file. Unfinished or broken: make it again. Only PyTorch missing: keep it, unless
rem PyTorch does not support its Python but supports a Python that is installed now.
if not exist "%NUMPY_OK%" goto renew_venv
"%VPY%" -c "import numpy" <nul >nul 2>&1
if errorlevel 1 goto renew_venv
if not errorlevel 0 goto renew_venv
call :find_python
if not defined PY_FITS goto install
"%VPY%" -c "import sys, sysconfig; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) and sysconfig.get_platform() == 'win-amd64' else 1)" <nul >nul 2>&1
if errorlevel 1 goto renew_venv
if not errorlevel 0 goto renew_venv
goto install

:renew_venv
call :find_python
if not defined PY goto no_python
echo Setting up the Python environment again: "%VENV%"
rem Its python.exe cannot be deleted while another battle assistant window uses it: then nothing else is.
if exist "%VPY%" del /f /q "%VPY%" >nul 2>&1
if exist "%VPY%" goto venv_locked
rmdir /s /q "%VENV%"
if exist "%VENV%" goto venv_locked
goto create

:existing_venv
rem Not created by this file, for example POKECHAMP_VENV: never deleted, the packages are installed into it.
"%VPY%" -c "import numpy, torch" <nul >nul 2>&1
if errorlevel 1 goto existing_venv_pip
if not errorlevel 0 goto existing_venv_pip
goto install

:existing_venv_pip
"%VPY%" -m pip --version <nul >nul 2>&1
if errorlevel 1 goto foreign_venv_broken
if not errorlevel 0 goto foreign_venv_broken
echo Using the existing Python environment "%VENV%"
goto install

:new_venv
call :find_python
if not defined PY goto no_python

:create
echo.
%PY% -c "import sys; print('Using Python', sys.version.split()[0], sys.executable)"
if not defined PY_FITS echo Note: PyTorch may not support this Python. Then the assistant starts without the neural network.
echo Creating the Python environment in "%VENV%" ...
%PY% -m venv "%VENV%"
if errorlevel 1 goto venv_create_failed
if exist "%VPY%" goto venv_created
rem Microsoft Store Python may hide what it writes under AppData from other programs: use this folder instead.
if /i "%VENV%"=="%HERE%.venv" goto venv_create_failed
set "VENV=%HERE%.venv"
set "OWN_DEFAULT="
goto check

:venv_created
echo created by run_ui.bat>"%TAG%"

:install
"%VPY%" -c "import numpy, torch" <nul >nul 2>&1
if errorlevel 1 goto need_packages
if not errorlevel 0 goto need_packages
goto deps_ok

:need_packages
if not exist "%NUMPY_OK%" goto install_numpy
"%VPY%" -c "import numpy" <nul >nul 2>&1
if errorlevel 1 goto install_numpy
if not errorlevel 0 goto install_numpy
rem numpy is there from an earlier start, so PyTorch failed then: do not wait long for it again
set "PIP_QUICK=--retries 1 --timeout 10"
goto install_torch

:install_numpy
echo.
echo Installing numpy. This happens only once.
"%VPY%" -m pip install --disable-pip-version-check numpy
if errorlevel 1 goto pip_failed
"%VPY%" -c "import numpy" <nul
if errorlevel 1 goto import_failed
if not errorlevel 0 goto import_failed
echo ok>"%NUMPY_OK%"

:install_torch
echo.
echo Installing PyTorch for the neural network AI: about 200 MB download, a few minutes.
"%VPY%" -m pip install --disable-pip-version-check %PIP_QUICK% torch
if errorlevel 1 goto torch_install_failed
"%VPY%" -c "import torch" <nul
if errorlevel 1 goto torch_import_failed
if not errorlevel 0 goto torch_import_failed

:deps_ok
echo ok>"%NUMPY_OK%"
echo ok>"%MARKER%"
echo Installation finished.

:start
echo.
"%VPY%" -m pokechamp ui --open %NO_MODEL% %*
if errorlevel 4 goto server_failed
if errorlevel 3 goto already_running
if errorlevel 1 goto server_failed
if not errorlevel 0 goto server_failed
endlocal
exit /b 0

:already_running
rem Exit code 3: it already runs in another window and its page was opened. Keep the message readable.
timeout /t 8 >nul 2>&1
endlocal
exit /b 0

rem ---------------------------------------------------------------------------------------------------------
:find_python
rem Prefers a Python that PyTorch supports, 64-bit x64 Python 3.10 to 3.14, then any Python 3.10 or newer.
rem Sets PY to the command, and PY_FITS when PyTorch supports that Python.
set "PY="
set "PY_FITS="
call :try_python 14 py -3
for %%V in (3.13 3.12 3.14 3.11 3.10) do if not defined PY call :try_python 14 py -%%V
if not defined PY call :try_python 14 python
if defined PY set "PY_FITS=1"
if not defined PY call :try_python 99 py -3
if not defined PY call :try_python 99 python
goto :eof

:try_python
rem Arguments: the highest accepted minor version, then the command. Sets PY when that Python fits.
rem Below 99 it must also be a 64-bit x64 Python, win-amd64: PyTorch has no 32-bit or ARM64 Windows build.
%2 %3 -c "import sys, sysconfig; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, %1) and (%1 == 99 or sysconfig.get_platform() == 'win-amd64') else 1)" <nul >nul 2>&1
if errorlevel 1 goto :eof
if not errorlevel 0 goto :eof
set "PY=%2 %3"
goto :eof

rem ---------------------------------------------------------------------------------------------------------
:torch_install_failed
echo.
echo [NOTE] PyTorch could not be installed. See the messages above.
goto no_torch

:torch_import_failed
echo.
echo [NOTE] PyTorch was installed but cannot be loaded. See the messages above.
echo   If the message mentions a DLL, install the Microsoft Visual C++ Redistributable from
echo   https://aka.ms/vs/17/release/vc_redist.x64.exe and run this file again.

:no_torch
rem No deps-ok marker is written, so the next start tries PyTorch again. pip stops quickly when there is none.
set "NO_MODEL=--no-model"
echo   The battle assistant now starts WITHOUT the neural network. It uses the heuristic AI, which is weaker.
"%VPY%" -c "import sys, sysconfig, platform; print('  This Python:', sys.version.split()[0], sysconfig.get_platform(), platform.architecture()[0], 'on', platform.machine())"
echo   For the full AI:
echo   - Check the internet connection and run this file again. Every start tries PyTorch again.
"%VPY%" -c "import sys, sysconfig; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) and sysconfig.get_platform() == 'win-amd64' else 1)" <nul >nul 2>&1
if errorlevel 1 goto no_torch_python
if not errorlevel 0 goto no_torch_python
goto start

:no_torch_python
echo   - PyTorch needs 64-bit Python 3.10 to 3.14 for x64 Windows, win-amd64. There is none for 32-bit or
echo     ARM64 Windows Python. Install "Python 3.14.x or 3.13.x - Windows installer (64-bit)" from
echo     https://www.python.org/downloads/windows/ - on Windows 11 on ARM too, not the ARM64 installer.
if exist "%TAG%" echo     Then run this file again: it sets up the environment again with that Python.
if not exist "%TAG%" echo     This file did not create "%VENV%", so it does not delete it. Delete that folder yourself,
if not exist "%TAG%" echo     then run this file again.
goto start

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
echo   1. Open https://www.python.org/downloads/windows/ and download
echo      "Python 3.14.x or 3.13.x - Windows installer (64-bit)". A newer Python may not support PyTorch yet.
echo   2. In the installer, tick the box "Add python.exe to PATH", then click "Install Now".
echo   3. Run this file again.
goto fail

:not_a_venv
echo.
echo [ERROR] "%VENV%" exists but is not a Python environment. Move or delete it, then run this file again.
goto fail

:foreign_venv_broken
echo.
echo [ERROR] The Python environment "%VENV%" does not work: its Python or pip cannot run.
echo   This file did not create that folder, so it does not delete it. Delete or move it yourself,
echo   or set POKECHAMP_VENV to another folder, then run this file again.
goto fail

:venv_locked
echo.
echo [ERROR] Could not delete "%VENV%". Close other battle assistant windows, then run this file again.
goto fail

:venv_create_failed
rem The folder did not exist before this attempt: remove only what the failed command left behind.
if exist "%VENV%" rmdir /s /q "%VENV%"
echo.
echo [ERROR] Could not create the Python environment in "%VENV%".
echo   Install "Python 3.14.x or 3.13.x - Windows installer (64-bit)" from https://www.python.org/downloads/windows/
echo   and run this file again.
goto fail

:pip_failed
echo.
echo [ERROR] Installing numpy failed. See the messages above.
"%VPY%" -c "import sys, sysconfig, platform; print('  This Python:', sys.version.split()[0], sysconfig.get_platform(), platform.architecture()[0], 'on', platform.machine())"
echo   - Check the internet connection and run this file again.
echo   - With a very new, 32-bit or ARM64 Python, install "Python 3.14.x or 3.13.x - Windows installer (64-bit)" from
echo     https://www.python.org/downloads/windows/ and run this file again.
goto fail

:import_failed
echo.
echo [ERROR] numpy was installed but cannot be loaded. See the messages above.
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
