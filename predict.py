import json
import os
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/src/weights"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()

import numpy as np
print(f"[module] numpy loaded", flush=True)
import torch
print(f"[module] torch {torch.__version__} loaded, cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
from cog import BasePredictor, Input, Path
print(f"[module] cog loaded", flush=True)
from PIL import Image
print(f"[module] PIL loaded", flush=True)
sys.stdout.flush()

WEIGHTS_DIR = "/src/weights/deepforest-livestock"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.setup_error = None
        print(f"[setup] WEIGHTS_DIR={WEIGHTS_DIR}", flush=True)
        try:
            print(f"[setup] dir contents: {sorted(os.listdir(WEIGHTS_DIR))[:20]}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
        print(f"[setup] cuda: {torch.cuda.is_available()}", flush=True)
        sys.stdout.flush()

        print(f"[setup] importing deepforest... (t={time.time()-t0:.1f}s)", flush=True)
        sys.stdout.flush()
        try:
            from deepforest import main as df_main
            self.df_main = df_main
            print(f"[setup] deepforest imported", flush=True)
        except Exception as e:
            import traceback
            print(f"[setup] FATAL deepforest import: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"deepforest import failed: {e}"
            return

        print(f"[setup] loading model from local path... (t={time.time()-t0:.1f}s)", flush=True)
        sys.stdout.flush()
        self.model = None

        # Acha o weight file primeiro (independente do método de load)
        weight_file = None
        try:
            files = sorted(os.listdir(WEIGHTS_DIR))
            print(f"[setup] WEIGHTS_DIR files: {files}", flush=True)
            for fname in ("model.bin", "pytorch_model.bin", "model.safetensors", "model.pth", "model.pt"):
                p = os.path.join(WEIGHTS_DIR, fname)
                if os.path.exists(p):
                    weight_file = p
                    break
        except Exception as e:
            print(f"[setup] erro listando weights: {e}", flush=True)

        if weight_file is None:
            self.setup_error = f"Nenhum weight file encontrado em {WEIGHTS_DIR}"
            print(f"[setup] FATAL: {self.setup_error}", flush=True)
            return

        print(f"[setup] weight file: {weight_file}", flush=True)
        # Carrega state_dict
        try:
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(weight_file)
            else:
                sd = torch.load(weight_file, map_location="cpu", weights_only=False)
            print(f"[setup] state_dict type: {type(sd)} | keys sample: {list(sd.keys())[:5] if isinstance(sd,dict) else 'tensor'}", flush=True)
            state = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        except Exception as e:
            self.setup_error = f"falha ao carregar state_dict: {e}"
            print(f"[setup] FATAL: {self.setup_error}", flush=True)
            import traceback; traceback.print_exc()
            sys.stdout.flush()
            return

        # Instancia deepforest sem trigger de download HF Hub
        # df_main.deepforest() construtor tenta baixar checkpoint default — patch isso
        try:
            # Constrói o RetinaNet torchvision diretamente (bypass do deepforest's default download)
            print(f"[setup] building RetinaNet via torchvision...", flush=True)
            from torchvision.models.detection import retinanet_resnet50_fpn_v2
            from torchvision.models.detection.retinanet import RetinaNetHead
            from torchvision.models.detection.anchor_utils import AnchorGenerator

            # Recupera num_classes do state_dict (head shape)
            num_classes_inferred = 2  # default = 1 class + background
            for k, v in state.items():
                if "classification_head.cls_logits.weight" in k or "head.classification.weight" in k:
                    # shape: [num_anchors*num_classes, in_channels, ...]
                    num_classes_inferred = max(num_classes_inferred, v.shape[0] // 9)  # 9 anchors típico
                    print(f"[setup] inferred num_classes={num_classes_inferred} from {k} shape={list(v.shape)}", flush=True)
                    break

            tv_model = retinanet_resnet50_fpn_v2(weights=None, weights_backbone=None, num_classes=num_classes_inferred)
            print(f"[setup] torchvision RetinaNet OK", flush=True)

            # Load state_dict no torchvision model
            missing, unexpected = tv_model.load_state_dict(state, strict=False)
            print(f"[setup] state_dict load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)
            if missing[:3]: print(f"  missing: {missing[:3]}", flush=True)
            if unexpected[:3]: print(f"  unexpected: {unexpected[:3]}", flush=True)

            # Cria wrapper deepforest VAZIO sem trigger download
            # Monkey-patch antes de instanciar
            print(f"[setup] criando deepforest wrapper...", flush=True)
            self.model = df_main.deepforest()
            self.model.model = tv_model  # substitui o backbone interno
            print(f"[setup] deepforest wrapper OK", flush=True)
        except Exception as e2:
            import traceback
            print(f"[setup] build FAILED: {type(e2).__name__}: {e2}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"build failed: {e2}"
            return

        self.model.eval()
        if torch.cuda.is_available():
            self.model.model = self.model.model.cuda()
        print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)
        sys.stdout.flush()

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
        if self.model is None:
            return json.dumps({"error": f"Modelo não carregou: {getattr(self, 'setup_error', 'unknown')}"})
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
