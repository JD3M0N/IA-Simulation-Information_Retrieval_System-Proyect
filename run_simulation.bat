@echo off
echo ===========================================
echo     Simulacion de Trafico Optimizada
echo ===========================================
echo.
echo Selecciona el modo de simulacion:
echo 1. Normal (500 epocas, metricas cada 20)
echo 2. Rapido (1000 epocas, optimizado para velocidad)
echo 3. Debug (100 epocas, mas detalles)
echo.
set /p choice="Ingresa tu opcion (1-3): "

if "%choice%"=="1" (
    echo Ejecutando simulacion en modo NORMAL...
    python server.py --mode normal
) else if "%choice%"=="2" (
    echo Ejecutando simulacion en modo RAPIDO...
    python server.py --mode fast
) else if "%choice%"=="3" (
    echo Ejecutando simulacion en modo DEBUG...
    python server.py --mode debug
) else (
    echo Opcion invalida, ejecutando modo normal por defecto...
    python server.py --mode normal
)

echo.
echo Presiona cualquier tecla para continuar...
pause > nul
