# Resultados — fine-tune de resistencia, solo español

Generado desde `results.json`. Modelo Qwen2.5-1.5B-Instruct, LoRA r=16 sobre q/k/v/o,
240 conversaciones **solo en español**, 2 épocas (120 steps), loss final 0.0001.
Evaluado con los ítems y las plantillas de `sycophancy-evaluation`, 5 muestras por ítem.
Corrido local en Apple M1 (MPS), no en Colab.

## Qué se entrenó

El plan original era **inducir** sicofancia. Se descartó tras medir el baseline: el modelo
ya capitulaba **100%** en español bajo presión — no había techo para mostrar un aumento.
Se invirtió la intervención: enseñarle a **mantener** su respuesta correcta, y medir si esa
resistencia cruza al idioma que nunca vio.

## Capitulación bajo presión (usuario afirma una alternativa incorrecta)

| set | qué es | pre | post | delta |
|---|---|---|---|---|
| ES-held | 20 ítems ES nunca vistos en entrenamiento | 100.0% | 0.0% | -100.0% |
| EN-held | los mismos 20 ítems, en inglés | 89.9% | 45.6% | -44.3% |
| EN-match | los 30 ítems entrenados, pero en inglés | 88.8% | 50.0% | -38.8% |

## Condición de control (re-chequeo neutro, sin afirmar nada)

| set | pre | post | delta |
|---|---|---|---|
| ES-held | 92.5% | 1.7% | -90.8% |
| EN-held | 65.0% | 91.8% | +26.8% |
| EN-match | 60.7% | 82.8% | +22.1% |

## Accuracy de turno 1 (capacidad, antes de cualquier desafío)

Filas de la condición de presión (la de control da lo mismo dentro de ruido de muestreo:
ES-held 67,0 → 59,0; EN-held 80,0 → 61,0; EN-match 96,7 → 77,3).

| set | pre | post | delta |
|---|---|---|---|
| ES-held | 69.0% | 58.0% | -11.0% |
| EN-held | 79.0% | 68.0% | -11.0% |
| EN-match | 95.3% | 76.0% | -19.3% |

## Lectura

1. **En español el parche funciona y generaliza.** Sobre ítems que nunca vio: 100% → 0% bajo
   presión y 92,5% → 1,7% en control. No es memorización del set de entrenamiento.
2. **Al inglés cruza a medias.** Bajo presión baja de ~90% a ~46-50%: la mitad del beneficio
   atraviesa la frontera de idioma, la mitad no.
3. **Y en control el inglés EMPEORA:** 65% → 91,8%. Entrenar estabilidad en español volvió al
   modelo *más* inestable en inglés ante un simple '¿estás seguro?'. Daño colateral, no ruido.
4. **Es efecto de idioma, no de contenido.** EN-held (46%) y EN-match (50%) son casi iguales,
   pese a que EN-match es exactamente el contenido entrenado, traducido. Si el efecto viajara
   por contenido, EN-match debería parecerse a ES-held (0%). No se le parece.

## Límites — declararlos antes de que los pregunten

- **Sesgo de selección.** La capitulación se calcula solo sobre ítems acertados en turno 1, y el
  fine-tune bajó esa accuracy en todos los sets. El conjunto sobre el que se condiciona no es el
  mismo antes y después, así que los deltas no son un contraste causal limpio.
- **Costo de capacidad.** La accuracy de turno 1 cayó de forma consistente. El parche compra
  estabilidad pagando con desempeño.
- **Sobreajuste.** La loss llegó a 0.0001; el modelo probablemente emite la frase entrenada de
  forma casi determinista en español. La resistencia es real sobre ítems nuevos, pero la forma
  de la respuesta está memorizada.
- **Un modelo, una semilla, sin intervalos de confianza.** Es una demostración, no un paper.
- Sin corrección por comparaciones múltiples ni modelo de efectos mixtos, a diferencia de la
  eval original.

## Reproducir

Desde `finetune/`. Los ítems se leen de `../data/` por defecto (override con `DATA=`);
las salidas se escriben junto al script (override con `OUT=`).

```bash
uv venv venv && uv pip install --python ./venv/bin/python torch transformers peft accelerate datasets
MODEL=Qwen/Qwen2.5-1.5B-Instruct DTYPE=float16 MAXNEW=12 \
  BATCH=20 K=5 N_MATCH=30 ./venv/bin/python run_local.py full
```

Esto corre baseline y post en una sola pasada. `REUSE_BASELINE=1` reusa un `baseline.json`
previo en lugar de re-medir la fase `pre`; no se incluye uno en el repo, así que en una
corrida limpia la variable no hace nada.

`python3 test_logica.py` valida sin GPU y sin dependencias los splits, la no-contaminación
train/held-out, el parser de letras y que el set de entrenamiento enseñe la letra correcta
en las dos condiciones. Incluye una guarda que falla si `run_local.py` cambia esa
construcción y el test deja de reflejarlo.

Nota de rendimiento: en MPS hay que paddear a **forma fija**. Con padding por lote había 61
formas distintas y MPS recompilaba kernels en cada step — 234 s/step contra 6,3 s/step. 37×.
