@echo off
setlocal enabledelayedexpansion
title Smart-Bin Waste Segregation Assistant

echo ======================================================================
echo   RECYCLE AI: SMART-BIN SORTING ASSISTANT
echo   Author: AGGIBADALA VENKATA DURGA NAGA PRASAD
echo   Supervisor: Ms. G. Hema Sri
echo   Domain: Deep Learning & Computer Vision
echo ======================================================================
echo.

cd /d "%~dp0"

:: 1. Check if streamlit is directly recognized
where streamlit >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Streamlit on System PATH.
    echo [*] Launching Smart-Bin Dashboard...
    echo.
    streamlit run app.py
    goto end
)

:: 2. Check Anaconda environments (checking anaconda5 first since packages were installed there)
for %%D in (anaconda5 anaconda3 anaconda8 anaconda7 anaconda4 Miniconda3 miniconda) do (
    if exist "%USERPROFILE%\%%D\Scripts\activate.bat" (
        echo [*] Found Anaconda environment at %USERPROFILE%\%%D
        echo [*] Activating %USERPROFILE%\%%D ...
        call "%USERPROFILE%\%%D\Scripts\activate.bat"
        
        if exist "%USERPROFILE%\%%D\Scripts\streamlit.exe" (
            echo [*] Found Streamlit in %%D!
            echo [*] Launching Smart-Bin Dashboard at http://localhost:8501 ...
            echo.
            "%USERPROFILE%\%%D\Scripts\streamlit.exe" run app.py
            goto end
        ) else (
            python -m streamlit --version >nul 2>nul
            if !ERRORLEVEL! EQU 0 (
                echo [*] Launching via Python module in %%D...
                python -m streamlit run app.py
                goto end
            )
        )
    )
)

:: 3. Check python -m streamlit on generic PATH
python -m streamlit --version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo [*] Found Streamlit under Python environment.
    echo [*] Launching Smart-Bin Dashboard...
    echo.
    python -m streamlit run app.py
    goto end
)

echo [!] Could not automatically start Streamlit.
echo.
echo Please run:
echo   call C:\Users\srila\anaconda5\Scripts\activate
echo   cd C:\Users\srila\.gemini\antigravity\scratch\waste_segregation_smart_bin
echo   streamlit run app.py
echo.

:end
pause
