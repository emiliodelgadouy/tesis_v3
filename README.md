# tesis_v3

## MAMMO-CLIP

MAMMO-CLIP queda integrado como baseline externo, no como backbone Keras entrenable. El modelo oficial usa PyTorch y su propio preprocesamiento, por lo que mezclarlo dentro de `ModelBuilder` como si fuera un `keras.Model` seria fragil.

Uso sugerido desde un notebook:

```python
from src.notebook_api import *

model = load_mammoclip_model()
zs = predict_mammoclip_zero_shot(
    test_df,
    model=model,
    config=MammoClipZeroShotConfig(output_csv="outputs/mammo_clip_zero_shot.csv"),
    limit=16,  # quitar para correr todo el split
)
zs.head()
```

Si falta el paquete opcional:

```bash
pip install mammoclip
```

`mammoclip` tambien aparece en el catalogo de arquitecturas para documentacion, pero `resolve_backbone("mammoclip")` levanta un error explicando que debe usarse con `src.mammo_clip`.
