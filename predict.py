import json
import os
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/src/weights"

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from PIL import Image

WEIGHTS_DIR = "/src/weights/deepforest-livestock"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(WEIGHTS_DIR))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)

        print(f"[setup] importing deepforest... (t={time.time()-t0:.1f}s)", flush=True)
        from deepforest import main as df_main
        self.df_main = df_main

        print(f"[setup] loading model from local path... (t={time.time()-t0:.1f}s)", flush=True)
        # Carrega usando from_pretrained do PyTorchModelHubMixin
        self.model = df_main.deepforest.from_pretrained(WEIGHTS_DIR)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.model = self.model.model.cuda()
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)

    def predict(
        self,
        image: Path = Input(description="Imagem aérea RGB (jpg/png/tif) com gado/livestock."),
        patch_size: int = Input(
            description="Tamanho do patch de inferência. Imagens grandes serão divididas em patches.",
            default=400,
            ge=200,
            le=2000,
        ),
        patch_overlap: float = Input(
            description="Overlap entre patches (0.0 a 0.5).",
            default=0.15,
            ge=0.0,
            le=0.5,
        ),
        iou_threshold: float = Input(
            description="IoU threshold pra NMS (Non-Maximum Suppression).",
            default=0.15,
            ge=0.0,
            le=1.0,
        ),
        score_threshold: float = Input(
            description="Score mínimo pra considerar uma detecção válida (0.0 a 1.0).",
            default=0.3,
            ge=0.0,
            le=1.0,
        ),
        return_format: str = Input(
            description="'detections' (JSON com boxes) ou 'summary' (estatísticas).",
            default="detections",
            choices=["detections", "summary"],
        ),
    ) -> str:
        path_str = str(image)
        print(f"[predict] image={path_str} patch_size={patch_size}", flush=True)

        # DeepForest tem 2 modos: predict_image (imagem inteira) e predict_tile (imagem grande, com patches)
        # Pra robustez, usamos predict_tile que funciona pra ambos
        try:
            preds = self.model.predict_tile(
                raster_path=path_str,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                iou_threshold=iou_threshold,
            )
        except Exception as e:
            print(f"[predict] predict_tile falhou ({e}), tentando predict_image", flush=True)
            preds = self.model.predict_image(path=path_str)

        if preds is None or len(preds) == 0:
            return json.dumps({
                "n_detections": 0,
                "detections": [],
                "message": "Nenhuma detecção encontrada (imagem pode não conter gado ou estar fora da distribuição do modelo).",
            }, ensure_ascii=False)

        # Filtrar por score_threshold
        if "score" in preds.columns:
            preds = preds[preds["score"] >= score_threshold]

        # Detecções como lista de dicts
        detections = []
        for _, row in preds.iterrows():
            det = {
                "xmin": float(row.get("xmin", 0)),
                "ymin": float(row.get("ymin", 0)),
                "xmax": float(row.get("xmax", 0)),
                "ymax": float(row.get("ymax", 0)),
                "label": str(row.get("label", "Livestock")),
                "score": float(row.get("score", 0.0)),
            }
            detections.append(det)

        result = {
            "n_detections": len(detections),
            "model": "weecology/deepforest-livestock",
            "score_threshold": score_threshold,
            "patch_size": patch_size,
        }

        if return_format == "detections":
            result["detections"] = detections
        else:
            # Summary mode
            if detections:
                scores = [d["score"] for d in detections]
                widths = [d["xmax"] - d["xmin"] for d in detections]
                heights = [d["ymax"] - d["ymin"] for d in detections]
                areas = [w * h for w, h in zip(widths, heights)]
                result["score_stats"] = {
                    "mean": float(np.mean(scores)),
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                }
                result["box_stats"] = {
                    "mean_area_px2": float(np.mean(areas)),
                    "total_area_px2": float(np.sum(areas)),
                }

        return json.dumps(result, ensure_ascii=False)
