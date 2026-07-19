# tesis_v3

Modos de entrenamiento:

- `simple`, `full`, `patch`, `patch_hardneg`
- `abmil`
- `abmil_patch_hardneg`: transfiere backbone y proyección densa; usa una salida de bag nueva
- `abmil_patch_hardneg_guided`: además usa el clasificador patch por instancia para inicializar
  la atención, sin reutilizarlo como clasificador global

Para la transferencia patch → ABMIL, entrenar `patch_hardneg` con
`PATCH_HARDNEG.ALIGN_TO_BAG_GRID=True`. El modo guiado usa:

- `MIL.GUIDED_ATTENTION_TEMPERATURE` para controlar la concentración del prior;
- `MIL.GUIDED_ATTENTION_STRENGTH` para ponderar el prior frente al residuo aprendible.
