@echo off
set VENV_NAME=cs7180

:: Update pip
python -m pip install --upgrade pip

:: Create the virtual environment
python -m venv %VENV_NAME%

:: Activate the environment
call %VENV_NAME%\Scripts\activate

:: Install packages
pip install -r requirements.txt

echo Environment setup is complete.
pause