"""
DeepForest Livestock — Replicate predictor (standalone, sem df_main.deepforest()).

Bypass do `deepforest.main.deepforest` wrapper que sempre tenta baixar checkpoint
default do HF Hub no constructor. Aqui:
1. Carrega weights de /src/weights/deepforest-livestock/model.bin diretamente
2. Instancia torchvision RetinaNet com num_classes inferido do state_dict
3. predict_tile manual: divide imagem em patches, infere, agrega, NMS

Mesma semântica do deepforest.predict_tile mas sem dependência da lib em runtime.
"""
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
import torchvision
print(f"[module] torch {torch.__version__} loaded, cuda={torch.cuda.is_available()}", flush=True)
sys.stdout.flush()
from cog import BasePredictor, Input, Path
print(f"[module] cog loaded", flush=True)
from PIL import Image
print(f"[module] PIL loaded", flush=True)
sys.stdout.flush()

WEIGHTS_DIR = "/src/weights/deepforest-livestock"


def _generate_patches(W, H, patch_size, overlap):
    """Gera coordenadas (x, y) dos patches com overlap."""
    if W <= patch_size and H <= patch_size:
        return [(0, 0, W, H)]
    stride = int(patch_size * (1 - overlap))
    stride = max(stride, 1)
    coords = []
    y = 0
    while y < H:
        x = 0
        while x < W:
            x2 = min(x + patch_size, W)
            y2 = min(y + patch_size, H)
            x1 = max(0, x2 - patch_size)  # last patch alinha à direita
            y1 = max(0, y2 - patch_size)
            coords.append((x1, y1, x2, y2))
            if x2 >= W: break
            x += stride
        if y2 >= H: break
        y += stride
    # dedup
    return list(set(coords))


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.label_dict = {0: "Livestock"}  # default
        self.setup_error = None

        try:
            files = sorted(os.listdir(WEIGHTS_DIR))
            print(f"[setup] WEIGHTS_DIR files: {files}", flush=True)
        except Exception as e:
            print(f"[setup] cannot list WEIGHTS_DIR: {e}", flush=True)
            self.setup_error = str(e)
            return

        # Localiza weight file
        weight_file = None
        for fname in ("model.bin", "pytorch_model.bin", "model.safetensors", "model.pth", "model.pt"):
            p = os.path.join(WEIGHTS_DIR, fname)
            if os.path.exists(p):
                weight_file = p
                break
        if weight_file is None:
            self.setup_error = f"Nenhum weight file em {WEIGHTS_DIR}: {files}"
            print(f"[setup] FATAL: {self.setup_error}", flush=True)
            return

        print(f"[setup] weight file: {weight_file}", flush=True)
        sys.stdout.flush()

        # Carrega state_dict
        try:
            if weight_file.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(weight_file)
            else:
                sd = torch.load(weight_file, map_location="cpu", weights_only=False)
            state = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            # State pode estar prefixed com "model." (Lightning padrão)
            new_state = {}
            for k, v in state.items():
                k2 = k[6:] if k.startswith("model.") else k
                new_state[k2] = v
            state = new_state
            print(f"[setup] state_dict keys: {len(state)} | sample: {list(state.keys())[:5]}", flush=True)
        except Exception as e:
            import traceback
            print(f"[setup] state_dict load FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"state_dict load failed: {e}"
            return

        # Infere num_classes do classification head
        num_classes = 2
        for k, v in state.items():
            if "classification_head.cls_logits.weight" in k:
                # shape: [num_anchors_per_loc * num_classes, in_channels, 3, 3]
                # RetinaNet usa 9 anchors por location (3 scales × 3 ratios) — mas algumas variantes usam outros
                # Vamos detectar por inspeção: ver se shape[0] é múltiplo de 9
                first_dim = v.shape[0]
                for n_anch in (9, 6, 3):
                    if first_dim % n_anch == 0:
                        num_classes = first_dim // n_anch
                        print(f"[setup] inferred num_classes={num_classes} (n_anchors={n_anch}, head shape={list(v.shape)})", flush=True)
                        break
                break

        # Constrói RetinaNet
        try:
            print(f"[setup] building torchvision RetinaNet... (t={time.time()-t0:.1f}s)", flush=True)
            sys.stdout.flush()
            # DeepForest 2.x usa RetinaNet V1 (FPN ResNet50). V2 tem fpn extra_blocks com out_channels=256
            # mas o V1 tem out_channels=2048. Checkpoint do deepforest é treinado em V1.
            from torchvision.models.detection import retinanet_resnet50_fpn
            tv_model = retinanet_resnet50_fpn(
                weights=None, weights_backbone=None, num_classes=num_classes,
            )
            print(f"[setup] tv_model built. Loading state_dict (strict=False)...", flush=True)
            missing, unexpected = tv_model.load_state_dict(state, strict=False)
            print(f"[setup] load: {len(missing)} missing | {len(unexpected)} unexpected", flush=True)
            if missing[:5]: print(f"  missing: {missing[:5]}", flush=True)
            if unexpected[:5]: print(f"  unexpected: {unexpected[:5]}", flush=True)
            sys.stdout.flush()

            tv_model.eval()
            if torch.cuda.is_available():
                tv_model = tv_model.cuda()
            self.model = tv_model
            self.num_classes = num_classes
            print(f"[setup] DONE (t={time.time()-t0:.1f}s) | num_classes={num_classes}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] tv_model build FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"tv_model build failed: {e}"
            return

    def predict(
        self,
        image: Path = Input(description="Imagem aérea RGB (jpg/png/tif) com gado/livestock."),
        patch_size: int = Input(description="Tamanho do patch.", default=400, ge=200, le=2000),
        patch_overlap: float = Input(description="Overlap entre patches (0-0.5).", default=0.15, ge=0.0, le=0.5),
        iou_threshold: float = Input(description="IoU pra NMS.", default=0.15, ge=0.0, le=1.0),
        score_threshold: float = Input(description="Score mínimo (0-1).", default=0.3, ge=0.0, le=1.0),
        return_format: str = Input(default="detections", choices=["detections", "summary"]),
    ) -> str:
        if self.model is None:
            return json.dumps({"error": f"Modelo não carregou: {getattr(self, 'setup_error', 'unknown')}"})

        device = next(self.model.parameters()).device
        path_str = str(image)
        print(f"[predict] image={path_str} patch_size={patch_size}", flush=True)

        # Carrega imagem
        pil = Image.open(path_str).convert("RGB")
        W, H = pil.size
        img_arr = np.asarray(pil, dtype=np.float32) / 255.0  # [H, W, 3] em [0,1]
        print(f"[predict] imagem {W}x{H}", flush=True)

        # Gera patches
        patches = _generate_patches(W, H, patch_size, patch_overlap)
        print(f"[predict] {len(patches)} patches", flush=True)

        all_boxes = []
        all_scores = []
        all_labels = []

        with torch.inference_mode():
            for (x1, y1, x2, y2) in patches:
                patch = img_arr[y1:y2, x1:x2, :]  # [h, w, 3]
                ph, pw = patch.shape[:2]
                if pw < 32 or ph < 32:
                    continue  # patches degenerados
                # torchvision detection espera [C, H, W] float tensors
                tens = torch.from_numpy(patch.transpose(2, 0, 1)).contiguous().to(device)
                outputs = self.model([tens])  # list of dicts {boxes, scores, labels}
                out = outputs[0]
                boxes = out["boxes"].cpu().numpy()  # [N, 4] xyxy
                scores = out["scores"].cpu().numpy()
                labels = out["labels"].cpu().numpy()
                # Ajusta coords pro full image
                if len(boxes):
                    boxes[:, [0, 2]] += x1
                    boxes[:, [1, 3]] += y1
                    all_boxes.append(boxes)
                    all_scores.append(scores)
                    all_labels.append(labels)

        if not all_boxes:
            return json.dumps({
                "n_detections": 0, "detections": [],
                "image_size": [W, H], "patches": len(patches),
                "message": "Nenhuma detecção encontrada.",
            }, ensure_ascii=False)

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Filtra score
        mask = scores >= score_threshold
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        # NMS global (em CPU pra simplicidade)
        if len(boxes) > 0:
            keep = torchvision.ops.nms(
                torch.from_numpy(boxes).float(),
                torch.from_numpy(scores).float(),
                iou_threshold,
            ).numpy()
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        detections = []
        for b, s, l in zip(boxes, scores, labels):
            detections.append({
                "xmin": float(b[0]), "ymin": float(b[1]),
                "xmax": float(b[2]), "ymax": float(b[3]),
                "label": self.label_dict.get(int(l), f"class_{int(l)}"),
                "label_id": int(l),
                "score": float(s),
            })
        detections.sort(key=lambda d: -d["score"])

        result = {
            "n_detections": len(detections),
            "image_size": [W, H],
            "patches": len(patches),
            "patch_size": patch_size,
            "score_threshold": score_threshold,
            "iou_threshold": iou_threshold,
            "num_classes": self.num_classes,
        }
        if return_format == "detections":
            result["detections"] = detections
        else:
            if detections:
                _s = np.array([d["score"] for d in detections])
                _a = np.array([(d["xmax"]-d["xmin"])*(d["ymax"]-d["ymin"]) for d in detections])
                result["score_stats"] = {"mean": float(_s.mean()), "min": float(_s.min()), "max": float(_s.max())}
                result["box_stats"] = {"mean_area_px2": float(_a.mean()), "total_area_px2": float(_a.sum())}
        return json.dumps(result, ensure_ascii=False)
