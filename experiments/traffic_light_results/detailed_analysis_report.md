# Análisis del Impacto de Semáforos en el Comportamiento Vehicular

**Fecha del análisis:** 2025-07-08 10:38:50

## Resumen Ejecutivo

Este análisis evalúa el impacto de diferentes configuraciones de semáforos en el comportamiento vehicular utilizando simulaciones y análisis estadístico avanzado.

## Configuraciones Analizadas

### no_lights
- **Velocidad promedio:** 38.26 unidades
- **Tiempo de espera promedio:** 0.00 pasos
- **Paradas promedio:** 0.00
- **Eficiencia:** 100.00%
- **Sin semáforos activos**

### standard
- **Velocidad promedio:** 37.10 unidades
- **Tiempo de espera promedio:** 1.00 pasos
- **Paradas promedio:** 37.63
- **Eficiencia:** 80.29%
- **Semáforos activos:** 46
- **Tiempo de ciclo:** 40s
- **Zonas:** 3

### fast_cycle
- **Velocidad promedio:** 36.83 unidades
- **Tiempo de espera promedio:** 1.00 pasos
- **Paradas promedio:** 36.37
- **Eficiencia:** 79.88%
- **Semáforos activos:** 46
- **Tiempo de ciclo:** 20s
- **Zonas:** 3

### slow_cycle
- **Velocidad promedio:** 36.78 unidades
- **Tiempo de espera promedio:** 1.00 pasos
- **Paradas promedio:** 38.10
- **Eficiencia:** 80.20%
- **Semáforos activos:** 46
- **Tiempo de ciclo:** 60s
- **Zonas:** 3

### green_wave
- **Velocidad promedio:** 36.45 unidades
- **Tiempo de espera promedio:** 1.00 pasos
- **Paradas promedio:** 36.73
- **Eficiencia:** 80.66%
- **Semáforos activos:** 46
- **Tiempo de ciclo:** 40s
- **Zonas:** 5

### random
- **Velocidad promedio:** 36.27 unidades
- **Tiempo de espera promedio:** 1.00 pasos
- **Paradas promedio:** 31.32
- **Eficiencia:** 82.98%
- **Semáforos activos:** 46
- **Tiempo de ciclo:** 40s
- **Zonas:** 1

## Análisis Estadístico

### Comparaciones Bootstrap

#### Speeds

**no_lights_vs_standard:**
- Diferencia de medias: 1.1659
- Intervalo de confianza 95%: [1.1048, 1.2207]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_fast_cycle:**
- Diferencia de medias: 1.4353
- Intervalo de confianza 95%: [1.3752, 1.4968]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_slow_cycle:**
- Diferencia de medias: 1.4789
- Intervalo de confianza 95%: [1.4171, 1.5386]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_green_wave:**
- Diferencia de medias: 1.8160
- Intervalo de confianza 95%: [1.7548, 1.8748]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_random:**
- Diferencia de medias: 1.9934
- Intervalo de confianza 95%: [1.9341, 2.0535]
- p-valor: 0.0000
- Significativo: Sí

**standard_vs_fast_cycle:**
- Diferencia de medias: 0.2676
- Intervalo de confianza 95%: [0.2086, 0.3266]
- p-valor: 0.0000
- Significativo: Sí

**standard_vs_slow_cycle:**
- Diferencia de medias: 0.3118
- Intervalo de confianza 95%: [0.2490, 0.3714]
- p-valor: 0.0000
- Significativo: Sí

**standard_vs_green_wave:**
- Diferencia de medias: 0.6493
- Intervalo de confianza 95%: [0.5902, 0.7108]
- p-valor: 0.0000
- Significativo: Sí

**standard_vs_random:**
- Diferencia de medias: 0.8268
- Intervalo de confianza 95%: [0.7663, 0.8861]
- p-valor: 0.0000
- Significativo: Sí

**fast_cycle_vs_slow_cycle:**
- Diferencia de medias: 0.0450
- Intervalo de confianza 95%: [-0.0165, 0.1056]
- p-valor: 0.1700
- Significativo: No

**fast_cycle_vs_green_wave:**
- Diferencia de medias: 0.3833
- Intervalo de confianza 95%: [0.3195, 0.4438]
- p-valor: 0.0000
- Significativo: Sí

**fast_cycle_vs_random:**
- Diferencia de medias: 0.5615
- Intervalo de confianza 95%: [0.4996, 0.6237]
- p-valor: 0.0000
- Significativo: Sí

**slow_cycle_vs_green_wave:**
- Diferencia de medias: 0.3356
- Intervalo de confianza 95%: [0.2729, 0.3969]
- p-valor: 0.0000
- Significativo: Sí

**slow_cycle_vs_random:**
- Diferencia de medias: 0.5128
- Intervalo de confianza 95%: [0.4490, 0.5795]
- p-valor: 0.0000
- Significativo: Sí

**green_wave_vs_random:**
- Diferencia de medias: 0.1778
- Intervalo de confianza 95%: [0.1142, 0.2426]
- p-valor: 0.0000
- Significativo: Sí

#### Wait Times

**standard_vs_fast_cycle:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**standard_vs_slow_cycle:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**standard_vs_green_wave:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**standard_vs_random:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**fast_cycle_vs_slow_cycle:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**fast_cycle_vs_green_wave:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**fast_cycle_vs_random:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**slow_cycle_vs_green_wave:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**slow_cycle_vs_random:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

**green_wave_vs_random:**
- Diferencia de medias: 0.0000
- Intervalo de confianza 95%: [0.0000, 0.0000]
- p-valor: 2.0000
- Significativo: No

#### Stop Counts

**no_lights_vs_standard:**
- Diferencia de medias: -37.6653
- Intervalo de confianza 95%: [-41.0254, -34.4745]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_fast_cycle:**
- Diferencia de medias: -36.4544
- Intervalo de confianza 95%: [-39.1076, -33.7447]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_slow_cycle:**
- Diferencia de medias: -38.1272
- Intervalo de confianza 95%: [-41.6547, -34.6494]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_green_wave:**
- Diferencia de medias: -36.7057
- Intervalo de confianza 95%: [-40.2223, -33.1870]
- p-valor: 0.0000
- Significativo: Sí

**no_lights_vs_random:**
- Diferencia de medias: -31.3878
- Intervalo de confianza 95%: [-34.3176, -28.4070]
- p-valor: 0.0000
- Significativo: Sí

**standard_vs_fast_cycle:**
- Diferencia de medias: 1.2022
- Intervalo de confianza 95%: [-2.9882, 5.4916]
- p-valor: 0.5960
- Significativo: No

**standard_vs_slow_cycle:**
- Diferencia de medias: -0.4459
- Intervalo de confianza 95%: [-5.4604, 4.4994]
- p-valor: 0.8520
- Significativo: No

**standard_vs_green_wave:**
- Diferencia de medias: 0.8268
- Intervalo de confianza 95%: [-3.9776, 5.4967]
- p-valor: 0.7460
- Significativo: No

**standard_vs_random:**
- Diferencia de medias: 6.3572
- Intervalo de confianza 95%: [1.9099, 10.7569]
- p-valor: 0.0040
- Significativo: Sí

**fast_cycle_vs_slow_cycle:**
- Diferencia de medias: -1.7022
- Intervalo de confianza 95%: [-6.2196, 2.5000]
- p-valor: 0.4560
- Significativo: No

**fast_cycle_vs_green_wave:**
- Diferencia de medias: -0.3723
- Intervalo de confianza 95%: [-4.5022, 3.9981]
- p-valor: 0.8840
- Significativo: No

**fast_cycle_vs_random:**
- Diferencia de medias: 4.9915
- Intervalo de confianza 95%: [0.9816, 8.7153]
- p-valor: 0.0160
- Significativo: Sí

**slow_cycle_vs_green_wave:**
- Diferencia de medias: 1.3222
- Intervalo de confianza 95%: [-3.6462, 6.1631]
- p-valor: 0.5740
- Significativo: No

**slow_cycle_vs_random:**
- Diferencia de medias: 6.8202
- Intervalo de confianza 95%: [2.0336, 11.2789]
- p-valor: 0.0040
- Significativo: Sí

**green_wave_vs_random:**
- Diferencia de medias: 5.3562
- Intervalo de confianza 95%: [1.0473, 9.6981]
- p-valor: 0.0200
- Significativo: Sí

## Conclusiones Principales

1. **Configuración más eficiente:** no_lights (100.00% eficiencia)

2. **Configuración con mayor velocidad:** no_lights (38.26 unidades promedio)

3. **Impacto de semáforos en velocidad:** Reducción promedio del 4.1%

