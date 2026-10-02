# Attack-anchored seed analysis

## Objetivo

Verificar auditivamente si las coincidencias detectadas por el índice de
`Block Rockin' Beats` corresponden realmente a samples reutilizables antes de
aplicar división local o extensión adaptativa.

## Enfoque

1. Se conserva el tempo detectado como prior estructural: `108.45 BPM`, con
   periodicidad dominante de `72.3 BPM`.
2. Los ataques se mantienen en su posición original. El tempo no se usa para
   cuantizar ni desplazar eventos fuera de grilla.
3. Cada ataque genera una semilla de `600 ms`, comenzando exactamente en el
   ataque.
4. La firma compara la secuencia STFT ordenada completa de la ventana: energía,
   evolución temporal, bandas espectrales y relaciones armónicas. No se usa
   espectro promedio temporal.
5. Se recuperan candidatos globalmente y se validan con FFT `256/1024/4096`,
   alineación acotada y métricas locales de tiempo-frecuencia.
6. Solo después de esta escucha de control se aplicará la extensión cuadro a
   cuadro y la división de intervalos incompatibles.

## Resultado de la pasada

| Métrica | Resultado |
|---|---:|
| Ataques/eventos indexados | 964 |
| Ventanas de semilla | 964 |
| Pares candidatos recuperados | 3.856 |
| Pares aceptados por validación completa | 16 |
| Bundles no superpuestos seleccionados | 12 |
| Cobertura temporal de bundles | 4,35 % |
| Energía reemplazada | 2,50 % |
| Error de descomposición | 0 |

La extensión adaptativa de `120 ms` produjo una cobertura de `4,35 %`, frente
a `3,48 %` sin extensión. No se interpreta como reconstrucción final: solo
mide cuánto contexto adicional sobrevive a la validación.

## Controles auditivos

Para cada uno de los 16 pares aceptados se exportaron tres archivos:

- `target`: ventana original en la posición del ataque.
- `candidate-raw`: ventana candidata sin modificar.
- `candidate-aligned`: candidata con la ganancia y el desplazamiento validados.

Los archivos están en `seeds/`, junto con `seeds/index.json`, que relaciona
cada archivo con sus tiempos, score, ganancia y lag. Estos controles no
contienen residual, extensión adaptativa ni división local.

## Interpretación

Un par solo será considerado una semilla válida si el candidato conserva la
evolución completa de la ventana, no solamente el ataque. La escucha debe
distinguir entre:

- mismo sample o transformación plausible;
- ataque parecido pero cola distinta;
- coincidencia causada por batería o energía transitoria;
- falso positivo producido por mezcla densa.

La siguiente etapa queda deliberadamente pausada hasta revisar estos controles.
