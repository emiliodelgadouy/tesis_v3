# tesis_v3

Modos de entrenamiento:

- `simple`, `full`, `patch`, `patch_hardneg`, `patch_alltiles`
- `multibranch`: fusiona una rama global FULL con una rama local ABMIL sobre el mismo canvas
- `abmil`
- `abmil_patch_hardneg`: transfiere backbone y proyección densa; usa una salida de bag nueva
- `abmil_patch_hardneg_guided`: además usa el clasificador patch por instancia para inicializar
  la atención, sin reutilizarlo como clasificador global
- `abmil_patch_alltiles`: transfer simple desde el preentrenamiento sobre candidatos de grilla
- `abmil_patch_alltiles_gated`: añade los logits patch con una compuerta aprendible
- `abmil_patch_alltiles_score`: concatena el logit patch al embedding de cada instancia

`patch_alltiles` etiqueta primero todos los tiles de la grilla compartida y en
train submuestrea por separado negativos de mamografías positivas y negativas.
Validación/test conservan los tiles exhaustivos. Sus ratios se configuran en
`PATCH_ALLTILES` y el notebook mínimo de ablación es
`transfer_alltiles_ablation.ipynb` en el workspace conductor.

Para la transferencia patch → ABMIL, entrenar `patch_hardneg` con
`PATCH_HARDNEG.ALIGN_TO_BAG_GRID=True`. El modo guiado usa:

- `MIL.GUIDED_ATTENTION_TEMPERATURE` para controlar la concentración del prior;
- `MIL.GUIDED_ATTENTION_STRENGTH` para ponderar el prior frente al residuo aprendible.

Antes de transferir el modelo patch se puede comprobar si la guía localiza el
hallazgo sobre los tiles reales:

```python
guide_diagnostics = evaluate_patch_guide_localization(
    CONFIG,
    patch_builder,
    val=val,
    test=test,
)
guide_diagnostics.summary
guide_diagnostics.show_examples("test")
```

La ROI no se usa para elegir tiles ni para predecir: solamente para medir
`roi_top1_hit_rate`, `roi_top3_hit_rate` y el rango del primer tile que toca la
anotación.
