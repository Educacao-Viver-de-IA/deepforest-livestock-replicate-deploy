# deepforest-livestock

Detector de gado em imagens aéreas RGB. Deploy do [weecology/deepforest-livestock](https://huggingface.co/weecology/deepforest-livestock) via Cog + GitHub Actions.

- **Modelo**: RetinaNet fine-tunado pra detecção de Livestock
- **Licença**: MIT (comercial OK)
- **Lib**: `deepforest==2.1.0`

## API

| Input | Default | Descrição |
|---|---|---|
| `image` | obrigatório | RGB aérea (jpg/png/tif) |
| `patch_size` | 400 | Tamanho de patch (200-2000) |
| `patch_overlap` | 0.15 | Overlap (0-0.5) |
| `iou_threshold` | 0.15 | NMS IoU |
| `score_threshold` | 0.3 | Score mínimo |
| `return_format` | "detections" | "detections" ou "summary" |

## Output

```json
{
  "n_detections": 42,
  "detections": [
    {"xmin": 120.5, "ymin": 80.3, "xmax": 180.2, "ymax": 140.7, "label": "Livestock", "score": 0.87},
    ...
  ]
}
```
