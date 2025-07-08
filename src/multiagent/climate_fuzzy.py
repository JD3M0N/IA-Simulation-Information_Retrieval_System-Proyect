
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# 1) Definición de universos de variables
# ------------------------------------------------
# Temperatura en °C: de 0 a 40
x_temp = np.arange(0, 41, 1)
# Humedad en %: de 0 a 100
x_hum  = np.arange(0, 101, 1)
# Velocidad de viento en km/h: de 0 a 100
x_wind = np.arange(0, 101, 1)
# Precipitación en mm/h: de 0 a 20
x_rain = np.arange(0, 21, 1)
# Riesgo (índice): de 0 a 10
x_risk = np.arange(0, 11, 1)

# 2) Creación de objetos difusos (antecedentes y consecuente)
# ------------------------------------------------
temp   = ctrl.Antecedent(x_temp, 'temperatura')
hum    = ctrl.Antecedent(x_hum,   'humedad')
wind   = ctrl.Antecedent(x_wind,  'viento')
rain   = ctrl.Antecedent(x_rain,  'precipitacion')
risk   = ctrl.Consequent(x_risk,  'riesgo')

# 3) Asignación de funciones de membresía
# ------------------------------------------------
# Temperatura: frío, templado, calor
temp['frio']     = fuzz.trimf(x_temp, [0, 0, 15])
temp['templado'] = fuzz.trimf(x_temp, [10, 20, 30])
temp['calor']    = fuzz.trimf(x_temp, [25, 40, 40])

# Humedad: seca, confort, húmeda
hum['seca']    = fuzz.trimf(x_hum, [0,   0,  40])
hum['confort'] = fuzz.trimf(x_hum, [30,  50,  70])
hum['humeda']  = fuzz.trimf(x_hum, [60, 100, 100])

# Viento: calma, brisa, fuerte
wind['calma']  = fuzz.trimf(x_wind, [0,   0,  30])
wind['brisa']  = fuzz.trimf(x_wind, [20,  40,  60])
wind['fuerte'] = fuzz.trimf(x_wind, [50, 100, 100])

# Precipitación: ninguna, ligera, fuerte
rain['ninguna'] = fuzz.trimf(x_rain, [0,  0,  5])
rain['ligera']  = fuzz.trimf(x_rain, [3,  8, 12])
rain['fuerte']  = fuzz.trimf(x_rain, [10, 20, 20])

# Riesgo: bajo, medio, alto
risk['bajo']   = fuzz.trimf(x_risk, [0, 0, 5])
risk['medio']  = fuzz.trimf(x_risk, [3, 5, 8])
risk['alto']   = fuzz.trimf(x_risk, [6,10,10])

# 4) Definición de reglas difusas
# ------------------------------------------------
reglas = [
    # Si hace calor Y humedad es alta → riesgo alto
    ctrl.Rule(temp['calor'] & hum['humeda'], riesgo['alto']),
    # Si frío O viento fuerte → riesgo medio
    ctrl.Rule(temp['frio'] | wind['fuerte'], riesgo['medio']),
    # Si humedad seca Y sin lluvia → riesgo bajo
    ctrl.Rule(hum['seca'] & rain['ninguna'], riesgo['bajo']),
    # Si lluvia fuerte → riesgo alto
    ctrl.Rule(rain['fuerte'], riesgo['alto']),
    # Si brisa suave Y templado → riesgo bajo
    ctrl.Rule(wind['brisa'] & temp['templado'], riesgo['bajo']),
]

# 5) Sistema de control y simulador
# ------------------------------------------------
sistema_control = ctrl.ControlSystem(reglas)
simulador      = ctrl.ControlSystemSimulation(sistema_control)

def analizar_clima(temperatura, humedad, viento, precipitacion):
    """
    Recibe valores numéricos de temperatura (°C), humedad (%),
    viento (km/h) y precipitación (mm/h), y devuelve
    un índice de riesgo defuzzificado (0 a 10).
    """
    # Mostrar los valores de entrada
    print(f"[DEBUG] Entradas - Temp: {temperatura}°C, Hum: {humedad}%, "
          f"Viento: {viento}km/h, Lluvia: {precipitacion}mm/h")

    # Cargar entradas en el simulador
    simulador.input['temperatura']    = temperatura
    simulador.input['humedad']        = humedad
    simulador.input['viento']         = viento
    simulador.input['precipitacion']  = precipitacion

    # Ejecutar inferencia difusa
    simulador.compute()

    # Obtener resultado defuzzificado
    riesgo_crisp = simulador.output['riesgo']
    print(f"[DEBUG] Riesgo difuso defuzzificado: {riesgo_crisp:.2f}/10")

    return riesgo_crisp
