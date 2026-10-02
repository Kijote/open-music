# Banco de filtros convolucionales para componentes musicales

## Motivación

Las semillas confirmadas son correctas, pero contienen contaminación de otras capas de la mezcla. Comparar el bloque completo contra el residuo produce falsos negativos. Una máscara única aprendida de una semilla también puede sobreajustarse a una realización concreta.

El siguiente enfoque usa un banco de filtros convolucionales generales derivados de las semillas confirmadas.

## Representación

La entrada es una STFT ordenada temporalmente:

- cuadros consecutivos sin promediar el tiempo;
- bandas espectrales conservadas;
- energía y envolvente por cuadro;
- relaciones armónicas;
- resolución múltiple para ataques, colas y estructuras largas.

No se usa un espectro promedio global.

## Banco de filtros

Cada semilla genera varios filtros complementarios:

1. **Ataque temporal**  
   Describe la subida rápida y el pico inicial.

2. **Envolvente**  
   Describe ataque, sustain y caída sin fijar una amplitud absoluta.

3. **Plantilla espectral**  
   Conserva la distribución de energía por bandas y armónicos.

4. **Evolución tiempo-frecuencia**  
   Conserva cómo se desplazan o modifican las bandas a lo largo de la ventana.

5. **Variantes transformadas**  
   Incluye pequeñas variaciones de ganancia, duración y tono.

Los filtros se aplican como correlaciones/matched filters sobre la STFT del tema y del residuo.

## Flujo

```
semillas confirmadas
→ extracción de patrones locales
→ banco de filtros convolucionales
→ aplicación al tema completo
→ mapas de activación
→ combinación de activaciones
→ candidato de componente
→ validación local
→ resta exacta
→ residuo siguiente
```

La máscara se aprende como familia de comportamientos, no como un único promedio de espectro.

## Validación

Una activación solo se acepta si conserva:

- ataque;
- envolvente;
- tono o relación armónica;
- evolución temporal;
- continuidad en los bordes;
- energía local;
- fase suficiente para evitar clicks.

La validación se realiza en la componente filtrada y luego se verifica sobre la mezcla completa.

## Resta iterativa

Para cada componente aceptada:

```
tema_0 = tema original
componente_1 = filtro_1(tema_0)
residuo_1 = tema_0 - componente_1

componente_2 = filtro_2(residuo_1)
residuo_2 = residuo_1 - componente_2
```

La conservación debe ser exacta:

```
tema = componentes_aceptadas + residuo_final
```

Nunca se descarta el residuo ni se reemplaza una región completa solo por similitud global.

## Restricciones

- No cuantizar ataques al tempo.
- No usar espectros promediados como firma final.
- No aceptar filtros por una sola métrica.
- No restar una activación sin validación local.
- Mantener trazabilidad de semilla, filtro, posición, transformación y score.

## Estado

Las 16 semillas ataque-ancladas fueron confirmadas auditivamente. La separación armónica global produjo 99 candidatos de learned filter, de los cuales solo 1 sobrevivió la validación armónica estricta. Esto motiva el banco de filtros multiescala y específico por comportamiento.
