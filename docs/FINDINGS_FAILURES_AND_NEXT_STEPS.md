# open-music — Hallazgos, errores y plan de avance

**Estado:** documento de ingeniería vivo  
**Fecha de corte:** 30 de septiembre de 2026  
**Casos estudiados:** `prehistoric-drum-loop`, `wip-loop`, `slowdrum`, *Stairway to Heaven* y *Block Rockin’ Beats*.

## 1. Objetivo del producto

`open-music` busca representar una obra musical como una combinación de:

- eventos temporales;
- módulos de audio reutilizables;
- patrones rítmicos y estructurales;
- capas continuas o residuales cuando no puedan expresarse como eventos discretos;
- transformaciones explícitas entre una instancia y un módulo reutilizable.

El objetivo no es comprimir PCM arbitrariamente ni reconstruir con un grano único por evento. La representación debe descubrir modularidad real, permitir resolver módulos contra bibliotecas de samples y reconstruir cualquier canción con pérdida controlada y explicable.

## 2. Reglas de evidencia adoptadas

Todo resultado debe clasificarse como:

- **Válido:** la métrica y la audición respaldan la conclusión.
- **Invalidado:** la prueba midió otra cosa, tuvo contaminación o dependió de una premisa incorrecta.
- **Hipótesis:** explicación plausible que todavía requiere una prueba aislada.

Ningún porcentaje global autoriza por sí solo la reutilización de un módulo. Una reconstrucción solamente se considera mejor cuando coinciden:

1. cobertura temporal;
2. continuidad sin clicks ni pops;
3. preservación tiempo-frecuencia;
4. residual no descriptivo musicalmente;
5. reducción real de módulos únicos;
6. validación auditiva A/B.

## 3. Hallazgos válidos

### 3.1 El residual puede contener dos cosas completamente diferentes

**Válido.** Se distinguieron:

- residuo estructural: cuerpo, colas, ambiente, componentes continuos o eventos faltantes;
- piso numérico: errores de redondeo sin contenido perceptual.

Amplificar indiscriminadamente el residual produjo ganancias artificiales de aproximadamente `10^16–10^17×` y convirtió el piso numérico en una señal engañosa. Se adoptó la categoría `numerical_floor` por debajo de `10^-12`, que no debe amplificarse ni convertirse en módulos.

### 3.2 Los eventos discretos no explican necesariamente toda la música

**Válido.** En material con notas sostenidas, ambiente, reverberación o texturas continuas, el residual puede ser una capa musical real. No debe suponerse que todo residual es error ni que toda señal debe convertirse en ataques aislados.

La arquitectura necesita soportar al menos:

- eventos discretos;
- streams o capas continuas;
- colas vinculadas a eventos;
- residual numérico descartable.

### 3.3 El tempo y la grilla ayudan a no perder ataques, pero no determinan el sample completo

**Válido.** En *Stairway to Heaven*, modelar el tempo —incluidos microajustes y eventos fuera de grilla— redujo significativamente los clicks y evitó perder ataques. La sincronía permaneció sólida durante la reconstrucción.

El tempo sirve como prior estructural y referencia temporal, no como cuantización destructiva. Los ataques reales pueden caer fuera de la grilla y deben conservarse.

### 3.4 La cobertura temporal y la reutilización son problemas distintos

**Válido.** En *Stairway to Heaven*, la reconstrucción de control sin reutilización alcanzó `99,47 %` de cobertura. Esto demostró que los intervalos podían cubrirse casi completamente.

Al reutilizar módulos, la energía explicada cayó a `92,08 %`. La diferencia de `7,39` puntos provino de sustituciones incorrectas, no de falta de cobertura temporal.

### 3.5 “Misma familia” no significa “mismo sample reemplazable”

**Válido.** En *Block Rockin’ Beats*, una clasificación tímbrica redujo 964 eventos activos a 118 familias y marcó 846 reutilizaciones. Sin embargo:

- correlación mediana de forma de onda: `0,1945`;
- energía PCM explicada: `18,094 %`;
- SNR del residual: `0,867 dB`.

La clasificación pudo reconocer tipos generales de sonido, pero no demostrar que una instancia pudiera sustituirse por otra.

### 3.6 La audición detectó falsos positivos que la métrica no detectaba

**Válido.** En el primer comparador del residual de *Stairway*, la mayoría de los pares sonaron muy parecidos, pero Andrés identificó espontáneamente los pares 17, 19 y 20 como diferentes aunque sus similitudes calculadas estaban cerca de `0,99`.

Una segunda métrica penalizó correctamente 19 y 20 principalmente por duración, pero el par 17 continuó cerca de `0,991`. Ese par es evidencia de que faltaban dimensiones perceptuales o que se habían comprimido excesivamente.

### 3.7 Reducir la ventana globalmente no es una solución suficiente

**Válido.** Fragmentar todo en unidades cada vez menores puede elevar artificialmente la fidelidad, pero:

- destruye notas largas y estructuras internas;
- aumenta el número de módulos;
- reduce o elimina la reutilización;
- aproxima el modelo a almacenamiento PCM granular;
- genera más bordes y riesgo de clicks.

La subdivisión debe ser adaptativa y guiada por error localizado.

## 4. Resultados por experimento

### 4.1 `prehistoric-drum-loop`

**Resultado válido como benchmark simple.**

- 8 ataques.
- 8 eventos.
- 4 módulos.
- `99,899 %` de energía explicada.
- SNR `29,97 dB`.
- Residuo estructural `0,1006 %`.

Auditivamente la reconstrucción fue prácticamente indistinguible. Al amplificar el residual apareció un “tomb” perceptible, señal de que una fracción pequeña puede conservar estructura musical aunque sea poco audible dentro de la mezcla.

### 4.2 `wip-loop` y `slowdrum`

**Resultado inicial invalidado como modelo modular completo.**

La capa “solo eventos” produjo `0 %` porque estos audios contenían material tonal o continuo que el detector de ataques no representaba.

El fallback granular recuperó prácticamente `100 %`, pero con reutilización cercana a `1,0` y almacenamiento aproximado de `2×`. Eso fue reconstrucción por copia fragmentada, no descubrimiento de modularidad.

### 4.3 Primera etapa de *Stairway to Heaven*

**Resultado invalidado.**

- 0 eventos detectados en una prueba inicial.
- reconstrucción modular silenciosa;
- fallback casi idéntico al original;
- 10.341 granos únicos para 10.341 eventos.

La fidelidad provenía de copiar el audio, no de modelarlo.

### 4.4 Ataques y flujo espectral en *Stairway*

**Resultado parcialmente válido.**

Una iteración detectó:

- 1.147 ataques;
- 1.123 módulos estrictos;
- solamente 8 módulos repetidos;
- `91,6 %` de energía en event-only;
- 12 eventos adicionales recuperados del residual;
- fallback de 5.171 granos y 887 módulos.

Esto mostró que el detector podía encontrar estructura, pero todavía producía demasiados módulos únicos y un residual musical importante.

### 4.5 Modelado rítmico de *Stairway*

**Resultado válido para cobertura y sincronía; inválido para reutilización.**

- 1.362 ataques conservados.
- 961 microataques.
- 301 submicroataques.
- 100 eventos fuera de grilla.
- 27 duplicados eliminados.
- 1.311 módulos.
- 34 módulos reutilizados.
- `99,47 %` cobertura de control sin reutilización.
- `92,08 %` energía explicada con reutilización.
- SNR aproximado `11,01 dB`.

Observaciones auditivas:

- tempo y sincronía sólidos;
- muchos menos clicks luego de incorporar microajustes;
- persistencia de clicks, pops y cortes;
- residual todavía descriptivo musicalmente;
- picos de envolvente entre módulos;
- módulos mayormente one-shot;
- ausencia de repetición estructural convincente.

### 4.6 Comparación de módulos residuales de *Stairway*

**Útil como experimento auditivo; métricas iniciales insuficientes.**

- 1.434 fragmentos temporales.
- 102 módulos con energía auditivamente relevante.
- 20 pares disjuntos seleccionados inicialmente.
- similitudes calculadas entre `0,98946` y `0,99431`.

Andrés consideró muy parecidos casi todos, excepto 17, 19 y 20. Esto probó que había reutilización potencial, pero también falsos positivos severos aun con valores cercanos a 1.

### 4.7 Primera pasada de *Block Rockin’ Beats*

**Tempo y segmentación preliminar válidos; familias y reutilización invalidadas.**

- duración analizada: `206,708 s`;
- sample rate: `44,1 kHz`, estéreo;
- periodicidad dominante detectada: `72,3 BPM`;
- relación métrica 3:2: `108,45 BPM`;
- 960 onsets;
- 965 eventos;
- 964 eventos activos;
- 118 familias;
- 78 familias repetidas;
- 846 eventos marcados para reutilización.

La cifra de familias queda invalidada porque la clasificación utilizó espectro promedio y compresión temporal excesiva.

## 5. Metidas de pata técnicas

### 5.1 Promediar el espectro

**Error crítico.** Se redujo la matriz tiempo-frecuencia `S(t,f)` a un promedio `S̄(f)`. Esto elimina el orden temporal y la variabilidad.

Dos bloques pueden tener el mismo promedio aunque uno evolucione `grave → agudo` y otro `agudo → grave`. El promedio no demuestra equivalencia ni reemplazabilidad.

**Decisión:** queda prohibido usar espectro promedio como criterio de aceptación de reemplazos.

### 5.2 Comprimir la evolución temporal a muy pocos bins

Se resumió la evolución en cuatro secciones y luego en ocho posiciones temporales. Esa reducción ocultó ataques desplazados, cambios de nota, silencios internos y variaciones de envolvente.

**Decisión:** conservar la secuencia tiempo-frecuencia completa. Los resúmenes solamente pueden servir para búsqueda aproximada de candidatos, nunca para aceptación.

### 5.3 Reducir el tono a un chroma global

El chroma global de 12 valores pierde:

- momento exacto de cada nota;
- octava;
- evolución armónica;
- parciales;
- transición entre notas.

Además, el peso tonal se reducía cuando el bloque parecía ruidoso. En mezclas densas, esto permitió aceptar reemplazos con notas diferentes.

**Decisión:** comparar tono, parciales y energía armónica por cuadro. Si la nota difiere, el módulo es distinto o requiere una transformación de pitch explícita.

### 5.4 Usar una ganancia global para corregir volumen

Una ganancia por canal no reproduce una envolvente variable. Dos bloques con RMS parecido pueden tener ataques, sostenidos y decaimientos completamente diferentes.

**Decisión:** comparar y, si corresponde, modelar una envolvente temporal explícita. Una transformación de amplitud debe registrarse como parámetro, no ocultarse en la sustitución.

### 5.5 Reemplazar bloques completos después de una comparación resumida

Se usó resolución fina para extraer descriptores, pero la decisión final reemplazaba todo el bloque. Una diferencia localizada contaminaba cientos de milisegundos.

**Decisión:** construir un mapa de error local y subdividir solamente las regiones incompatibles.

### 5.6 Agrupar contra un único ejemplar

El clustering online comparó cada evento con el ejemplar de la familia. Esto no garantiza consistencia entre todos los miembros y permite problemas transitivos: A≈B y B≈C no implica A≈C.

**Decisión:** toda familia reutilizable debe cumplir consistencia par a par o respecto de una envolvente de tolerancia común verificable.

### 5.7 Confundir similitud perceptual con igualdad PCM

Dos sonidos pueden ser auditivamente equivalentes pero tener fase distinta y producir un residual PCM grande. También pueden tener espectro global parecido y ser auditivamente diferentes.

**Decisión:** mantener métricas separadas:

- identidad PCM;
- equivalencia perceptual;
- equivalencia estructural;
- reemplazabilidad con transformación.

### 5.8 Usar una única métrica global

Un promedio permite que una dimensión excelente compense una falla grave en otra.

**Decisión:** usar restricciones locales independientes. Una diferencia tonal importante no puede compensarse con duración idéntica.

### 5.9 Interpretar energía explicada sin escuchar el residual

Un porcentaje alto puede esconder música claramente reconocible. Un residual de baja energía puede conservar melodía, ritmo o ataques descriptivos.

**Decisión:** toda prueba debe exportar residual a ganancia natural, residual amplificado con límite seguro y escucha A/B.

### 5.10 Confundir fallback granular con modularidad

Una reconstrucción casi exacta con un fragmento único por posición no prueba el producto. Solamente demuestra que los cortes cubren el audio.

**Decisión:** informar siempre por separado fidelidad, cantidad de módulos únicos, eventos, reutilización y costo de almacenamiento.

### 5.11 Tracker visual sobrecargado

Se intentó mostrar más de 800 módulos activos simultáneamente. La interfaz dejó de ser útil para inspección humana.

**Decisión:** vistas jerárquicas, virtualización, filtros por familia/error y zoom temporal. No renderizar todo con el mismo nivel de detalle.

### 5.12 Artefactos no verificados

Se entregaron HTML con errores como:

- `Invalid or unexpected token`;
- `Identifier 'top' has already been declared`;
- reproductores A/B que no funcionaban;
- enlaces de descarga que fallaban.

Algunos errores se repitieron después de haber sido reportados.

**Decisión:** ningún artefacto se entrega sin:

- parseo JavaScript;
- carga en navegador;
- consola sin errores;
- conteo de reproductores;
- reproducción de al menos un A/B;
- enlaces probados;
- archivos autocontenidos cuando sea posible.

### 5.13 Confusión entre source y regenerado

En una iteración, el regenerado resultó ser prácticamente el source completo mientras “solo eventos” estaba parcializado. Esto hizo que la aparente calidad de la reconstrucción fuera engañosa.

**Decisión:** cada salida debe declarar exactamente sus capas y demostrar procedencia mediante hashes, manifiesto de mezcla y prueba de anulación.

## 6. Arquitectura técnica corregida

### 6.1 Separar búsqueda de candidatos y validación

La búsqueda rápida puede usar embeddings compactos para reducir el universo de pares. No puede autorizar reemplazos.

La validación final debe comparar las matrices completas y alineadas:

1. candidato aproximado;
2. alineación temporal restringida;
3. comparación multirresolución;
4. mapa de error `E(t,f)`;
5. decisión local;
6. subdivisión o rechazo.

### 6.2 FFT multirresolución

Usar simultáneamente:

- FFT 4096 para graves, fundamentales y parciales;
- FFT 1024 para timbre y evolución intermedia;
- FFT 256 para ataques y microtransitorios.

Cada resolución conserva todos sus cuadros. No se promedia sobre el tiempo para validar.

### 6.3 Validación local por reemplazo

Para cada par objetivo/candidato medir por cuadro:

- diferencia log-espectral por banda;
- fundamental y chroma temporal;
- energía RMS;
- ataque, decaimiento y flujo espectral;
- coherencia estéreo y posición espacial;
- correlación con tolerancia pequeña de lag;
- continuidad en los bordes.

La decisión utiliza:

- percentil 95 del error;
- máximo error por cuadro;
- porcentaje de cuadros compatibles;
- máxima secuencia temporal incompatible;
- error tonal;
- error de envolvente;
- error de borde.

### 6.4 Subdivisión adaptativa

Si el error es localizado, dividir únicamente la región conflictiva. Ejemplo:

```text
bloque de 420 ms
├── 0–95 ms: ataque reutilizable
├── 95–270 ms: contenido tonal diferente
└── 270–420 ms: cola reutilizable
```

La recursión se detiene cuando:

- el bloque es validado;
- alcanza la resolución mínima;
- deja de producir reducción neta;
- separar aumentaría el error de bordes.

### 6.5 Reconstrucción

- ventanas Hann o equivalentes;
- overlap-add con normalización;
- transformaciones declaradas: ganancia, envolvente, pitch, stretch y posición estéreo;
- ninguna transformación inferida silenciosamente;
- prueba de continuidad en cada borde.

## 7. Métricas obligatorias

Cada corrida debe producir:

| Dimensión | Métrica |
|---|---|
| Cobertura | porcentaje temporal y energético antes de reutilizar |
| Modularidad | eventos / módulos únicos |
| Reutilización | eventos realmente reemplazados |
| Fidelidad PCM | SNR y energía residual |
| Fidelidad espectral | error tiempo-frecuencia por percentiles |
| Tono | cuadros con fundamental/chroma compatible |
| Dinámica | error de envolvente por cuadro |
| Continuidad | clicks, saltos de amplitud y energía de borde |
| Espacio | error de balance/correlación estéreo |
| Residual | clasificación estructural/continuo/numérico |
| Almacenamiento | bytes de módulos + eventos + transformaciones |

Ninguna métrica se presenta sin su control correspondiente.

## 8. Controles obligatorios

Cada evaluación de similitud debe mezclar de forma ciega:

- pares idénticos;
- el mismo módulo con ganancia diferente;
- el mismo módulo con pequeño desplazamiento;
- misma familia pero nota distinta;
- misma nota con ataque distinto;
- misma energía promedio con orden temporal invertido;
- pares claramente diferentes;
- candidatos reales del modelo.

Las etiquetas se revelan después de escuchar.

## 9. Plan rápido de implementación

### P0 — Invalidar y proteger

1. Eliminar espectro promedio del criterio de aceptación.
2. Marcar las 118 familias y 846 reutilizaciones de *Block Rockin’ Beats* como resultados inválidos.
3. Añadir tests que demuestren que invertir el orden temporal no conserva similitud.
4. Añadir tests contra compensación entre dimensiones.

### P1 — Validador par a par tiempo-frecuencia

1. STFT multirresolución completa.
2. Alineación restringida.
3. Error por cuadro y banda.
4. Tono y envolvente temporales.
5. Métricas locales sin promedio compensatorio.
6. Export de mapa de error y A/B.

**Criterio de salida:** los antiguos falsos positivos 17, 19 y 20 deben ser rechazados o explicados mediante una transformación explícita.

### P2 — Subdivisión adaptativa

1. Localizar intervalos incompatibles.
2. Dividir solamente esos intervalos.
3. Revalidar recursivamente.
4. Limitar profundidad y costo.
5. Reconstruir mediante overlap-add.

**Criterio de salida:** reducir residual musical sin multiplicar indiscriminadamente los módulos.

### P3 — Familias consistentes

1. Construir grafo de compatibilidad validada.
2. Exigir cliques o consistencia fuerte, no transitividad asumida.
3. Separar equivalencia exacta, perceptual y transformable.
4. Elegir ejemplar por mínimo error total, no por orden de llegada.

### P4 — Reconstrucción de *Block Rockin’ Beats*

1. Reconstrucción de control sin reutilización.
2. Reconstrucción reutilizada validada.
3. Residual natural y amplificado de forma segura.
4. Comparador ciego.
5. Medición de reducción de módulos y almacenamiento.

### P5 — Regresión sobre *Stairway to Heaven*

Repetir exactamente el mismo pipeline sin parámetros específicos por canción. *Stairway* sigue siendo el benchmark complejo para tempo variable, instrumentación múltiple, acordes, aceleraciones y ejecución humana.

## 10. Criterios de aceptación del siguiente milestone

El siguiente milestone no se considera completo hasta que:

- el sistema detecte aproximadamente 109 BPM en *Block Rockin’ Beats* sin hardcodear la canción;
- ninguna aceptación use espectro promedio temporal;
- cada reemplazo tenga un informe tiempo-frecuencia;
- los falsos positivos conocidos sean rechazados;
- la reconstrucción no introduzca clicks perceptibles;
- el residual deje de describir claramente la canción;
- la reutilización reduzca módulos y almacenamiento de manera medible;
- los resultados sean reproducibles desde un comando;
- HTML, JSON y audios pasen verificación automática antes de entregarse.

## 11. Principio de producto consolidado

> Un módulo no es una porción de audio que se parece en promedio a otra. Es una estructura temporal y espectral reutilizable cuya sustitución completa —o transformación explícita— ha sido validada localmente sin eliminar información musical relevante.

Este principio debe gobernar el modelado, las métricas, los tests y la interfaz de inspección.
