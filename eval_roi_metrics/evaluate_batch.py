import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


FRAME_RE = re.compile(r"^(\d{5})\.png$")


@dataclass
class MethodMetrics:
    dataset: str
    method: str
    bitrate_kbps: float
    mean_psnr: float
    mean_lpips: float
    mean_roi_psnr: float
    roi_valid_frames: int
    frame_count: int
    eval_frame_count: int
    payload_bytes: int
    video_seconds: float
    notes: str = ""


def _load_basnet_model(basnet_ckpt: Path, device: torch.device):
    basnet_root = basnet_ckpt.resolve().parents[2]
    if not basnet_root.exists():
        raise FileNotFoundError(f"BASNet root not found from ckpt path: {basnet_root}")
    sys.path.insert(0, str(basnet_root))
    from model import BASNet as BASNetModel  # pylint: disable=import-error

    net = BASNetModel(3, 1)
    state = torch.load(str(basnet_ckpt), map_location="cpu")
    net.load_state_dict(state)
    net = net.to(device).eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def _discover_datasets(baseline_root: Path, roi_root: Path) -> List[str]:
    base_data = baseline_root / "data"
    roi_data = roi_root / "data"
    if not base_data.exists() or not roi_data.exists():
        return []
    base_sets = {p.name for p in base_data.iterdir() if p.is_dir()}
    roi_sets = {p.name for p in roi_data.iterdir() if p.is_dir()}
    return sorted(base_sets.intersection(roi_sets))


def _list_frame_ids(folder: Path) -> List[int]:
    if not folder.exists():
        return []
    ids = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        m = FRAME_RE.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def _read_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _psnr_rgb(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def _preprocess_for_basnet(rgb01: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(rgb01, size=(256, 256), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _roi_psnr(
    basnet,
    gt_rgb: np.ndarray,
    pred_rgb: np.ndarray,
    threshold: float,
    device: torch.device,
) -> Tuple[Optional[float], bool]:
    gt01 = torch.from_numpy(gt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pred01 = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        roi_prob, *_ = basnet(_preprocess_for_basnet(gt01))
    roi_mask = (roi_prob >= threshold).float()
    roi_count = int(roi_mask.sum().item())
    if roi_count == 0:
        return None, False

    gt_256 = F.interpolate(gt01, size=(256, 256), mode="bilinear", align_corners=False) * 255.0
    pred_256 = F.interpolate(pred01, size=(256, 256), mode="bilinear", align_corners=False) * 255.0

    sq = (gt_256 - pred_256) ** 2
    sq_gray = sq.mean(dim=1, keepdim=True)
    mse_roi = (sq_gray * roi_mask).sum() / roi_mask.sum()
    mse_roi_v = float(mse_roi.item())
    if mse_roi_v <= 1e-12:
        return 99.0, True
    psnr_roi = 10.0 * math.log10((255.0 * 255.0) / mse_roi_v)
    return psnr_roi, True


def _payload_bytes(result_dir: Path) -> int:
    prompt_files = sorted(result_dir.glob("frame_*.prompt"))
    total = sum(p.stat().st_size for p in prompt_files)
    init_path = result_dir / "init.pth"
    if init_path.exists():
        total += init_path.stat().st_size
    return total


def _evaluate_method(
    dataset: str,
    method: str,
    gt_dir: Path,
    result_dir: Path,
    fps: float,
    roi_threshold: float,
    basnet,
    lpips_metric,
    device: torch.device,
    max_frames: Optional[int],
) -> MethodMetrics:
    gt_ids = _list_frame_ids(gt_dir)
    pred_ids = _list_frame_ids(result_dir)
    common_ids = sorted(set(gt_ids).intersection(pred_ids))
    if max_frames is not None:
        common_ids = common_ids[:max_frames]
    if len(common_ids) == 0:
        return MethodMetrics(
            dataset=dataset,
            method=method,
            bitrate_kbps=float("nan"),
            mean_psnr=float("nan"),
            mean_lpips=float("nan"),
            mean_roi_psnr=float("nan"),
            roi_valid_frames=0,
            frame_count=len(gt_ids),
            eval_frame_count=0,
            payload_bytes=0,
            video_seconds=0.0,
            notes="no overlapping frames",
        )

    psnr_vals: List[float] = []
    lpips_vals: List[float] = []
    roi_psnr_vals: List[float] = []
    roi_valid = 0

    for fid in common_ids:
        name = f"{fid:05d}.png"
        gt_rgb = _read_rgb(gt_dir / name)
        pred_rgb = _read_rgb(result_dir / name)
        if pred_rgb.shape != gt_rgb.shape:
            pred_rgb = cv2.resize(pred_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

        psnr_vals.append(_psnr_rgb(gt_rgb, pred_rgb))

        gt_t = torch.from_numpy(gt_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        pred_t = torch.from_numpy(pred_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            lp = lpips_metric(pred_t * 2.0 - 1.0, gt_t * 2.0 - 1.0).item()
        lpips_vals.append(float(lp))

        roi_psnr, valid = _roi_psnr(basnet, gt_rgb, pred_rgb, roi_threshold, device)
        if valid and roi_psnr is not None:
            roi_valid += 1
            roi_psnr_vals.append(float(roi_psnr))

    payload = _payload_bytes(result_dir)
    frame_count = len(gt_ids) if max_frames is None else min(len(gt_ids), max_frames)
    video_seconds = frame_count / fps if frame_count > 0 else 0.0
    bitrate_kbps = (8.0 * payload / video_seconds / 1000.0) if video_seconds > 0 else float("nan")

    return MethodMetrics(
        dataset=dataset,
        method=method,
        bitrate_kbps=float(bitrate_kbps),
        mean_psnr=float(np.mean(psnr_vals)),
        mean_lpips=float(np.mean(lpips_vals)),
        mean_roi_psnr=float(np.mean(roi_psnr_vals)) if roi_psnr_vals else float("nan"),
        roi_valid_frames=roi_valid,
        frame_count=frame_count,
        eval_frame_count=len(common_ids),
        payload_bytes=payload,
        video_seconds=float(video_seconds),
    )


def _to_row(m: MethodMetrics) -> Dict[str, object]:
    return {
        "dataset": m.dataset,
        "method": m.method,
        "bitrate_kbps": m.bitrate_kbps,
        "mean_psnr": m.mean_psnr,
        "mean_lpips": m.mean_lpips,
        "mean_roi_psnr": m.mean_roi_psnr,
        "roi_valid_frames": m.roi_valid_frames,
        "frame_count": m.frame_count,
        "eval_frame_count": m.eval_frame_count,
        "payload_bytes": m.payload_bytes,
        "video_seconds": m.video_seconds,
        "notes": m.notes,
    }


def _safe_delta(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return b - a


def _build_overall(rows: List[MethodMetrics]) -> List[Dict[str, object]]:
    by_dataset: Dict[str, Dict[str, MethodMetrics]] = {}
    for r in rows:
        by_dataset.setdefault(r.dataset, {})[r.method] = r

    table: List[Dict[str, object]] = []
    paired = []
    for d in sorted(by_dataset.keys()):
        x = by_dataset[d]
        if "baseline" in x and "roi_promptus" in x:
            b = x["baseline"]
            r = x["roi_promptus"]
            paired.append((b, r))
            table.append(
                {
                    "dataset": d,
                    "bitrate_kbps_baseline": b.bitrate_kbps,
                    "bitrate_kbps_roi": r.bitrate_kbps,
                    "delta_bitrate_kbps": _safe_delta(b.bitrate_kbps, r.bitrate_kbps),
                    "mean_psnr_baseline": b.mean_psnr,
                    "mean_psnr_roi": r.mean_psnr,
                    "delta_psnr": _safe_delta(b.mean_psnr, r.mean_psnr),
                    "mean_lpips_baseline": b.mean_lpips,
                    "mean_lpips_roi": r.mean_lpips,
                    "delta_lpips": _safe_delta(b.mean_lpips, r.mean_lpips),
                    "mean_roi_psnr_baseline": b.mean_roi_psnr,
                    "mean_roi_psnr_roi": r.mean_roi_psnr,
                    "delta_roi_psnr": _safe_delta(b.mean_roi_psnr, r.mean_roi_psnr),
                    "frame_count": min(b.frame_count, r.frame_count),
                }
            )

    if not paired:
        return table

    def macro(metric: str, idx: int) -> float:
        vals = []
        for b, r in paired:
            v = getattr((b, r)[idx], metric)
            if not math.isnan(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    weights = [min(b.frame_count, r.frame_count) for b, r in paired]
    total_w = sum(weights)

    def wavg(metric: str, idx: int) -> float:
        vals = []
        ws = []
        for (b, r), w in zip(paired, weights):
            v = getattr((b, r)[idx], metric)
            if not math.isnan(v):
                vals.append(v)
                ws.append(w)
        if not vals or sum(ws) == 0:
            return float("nan")
        return float(np.average(vals, weights=ws))

    macro_row = {
        "dataset": "__macro_avg__",
        "bitrate_kbps_baseline": macro("bitrate_kbps", 0),
        "bitrate_kbps_roi": macro("bitrate_kbps", 1),
        "delta_bitrate_kbps": float("nan"),
        "mean_psnr_baseline": macro("mean_psnr", 0),
        "mean_psnr_roi": macro("mean_psnr", 1),
        "delta_psnr": float("nan"),
        "mean_lpips_baseline": macro("mean_lpips", 0),
        "mean_lpips_roi": macro("mean_lpips", 1),
        "delta_lpips": float("nan"),
        "mean_roi_psnr_baseline": macro("mean_roi_psnr", 0),
        "mean_roi_psnr_roi": macro("mean_roi_psnr", 1),
        "delta_roi_psnr": float("nan"),
        "frame_count": total_w,
    }
    macro_row["delta_bitrate_kbps"] = _safe_delta(macro_row["bitrate_kbps_baseline"], macro_row["bitrate_kbps_roi"])
    macro_row["delta_psnr"] = _safe_delta(macro_row["mean_psnr_baseline"], macro_row["mean_psnr_roi"])
    macro_row["delta_lpips"] = _safe_delta(macro_row["mean_lpips_baseline"], macro_row["mean_lpips_roi"])
    macro_row["delta_roi_psnr"] = _safe_delta(macro_row["mean_roi_psnr_baseline"], macro_row["mean_roi_psnr_roi"])

    w_row = {
        "dataset": "__frame_weighted_avg__",
        "bitrate_kbps_baseline": wavg("bitrate_kbps", 0),
        "bitrate_kbps_roi": wavg("bitrate_kbps", 1),
        "delta_bitrate_kbps": float("nan"),
        "mean_psnr_baseline": wavg("mean_psnr", 0),
        "mean_psnr_roi": wavg("mean_psnr", 1),
        "delta_psnr": float("nan"),
        "mean_lpips_baseline": wavg("mean_lpips", 0),
        "mean_lpips_roi": wavg("mean_lpips", 1),
        "delta_lpips": float("nan"),
        "mean_roi_psnr_baseline": wavg("mean_roi_psnr", 0),
        "mean_roi_psnr_roi": wavg("mean_roi_psnr", 1),
        "delta_roi_psnr": float("nan"),
        "frame_count": total_w,
    }
    w_row["delta_bitrate_kbps"] = _safe_delta(w_row["bitrate_kbps_baseline"], w_row["bitrate_kbps_roi"])
    w_row["delta_psnr"] = _safe_delta(w_row["mean_psnr_baseline"], w_row["mean_psnr_roi"])
    w_row["delta_lpips"] = _safe_delta(w_row["mean_lpips_baseline"], w_row["mean_lpips_roi"])
    w_row["delta_roi_psnr"] = _safe_delta(w_row["mean_roi_psnr_baseline"], w_row["mean_roi_psnr_roi"])

    table.append(macro_row)
    table.append(w_row)
    return table


def _write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.6f}"
    return str(v)


def _write_md(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r[k]) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", type=str, required=True)
    parser.add_argument("--roi_root", type=str, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--interval", type=int, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--basnet_ckpt", type=str, required=True)
    parser.add_argument("--roi_threshold", type=float, default=0.5)
    parser.add_argument("--datasets", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="eval_roi_metrics/out")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root).resolve()
    roi_root = Path(args.roi_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets.strip():
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    else:
        datasets = _discover_datasets(baseline_root, roi_root)
    if not datasets:
        raise RuntimeError("No common datasets found under baseline_root/data and roi_root/data")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    basnet = _load_basnet_model(Path(args.basnet_ckpt), device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device).eval()

    rank_dir = f"rank{args.rank}_interval{args.interval}"
    max_frames = args.max_frames if args.max_frames > 0 else None
    rows: List[MethodMetrics] = []
    skipped: List[Dict[str, str]] = []

    for dataset in datasets:
        gt_dir = roi_root / "data" / dataset
        baseline_gt = baseline_root / "data" / dataset
        base_result = baseline_root / "data" / dataset / "results" / rank_dir
        roi_result = roi_root / "data" / dataset / "results" / rank_dir

        if not gt_dir.exists():
            skipped.append({"dataset": dataset, "reason": f"gt_dir_not_found: {gt_dir}"})
            continue
        if not baseline_gt.exists():
            skipped.append({"dataset": dataset, "reason": f"baseline_gt_not_found: {baseline_gt}"})
            continue
        if not base_result.exists():
            skipped.append({"dataset": dataset, "reason": f"baseline_result_not_found: {base_result}"})
            continue
        if not roi_result.exists():
            skipped.append({"dataset": dataset, "reason": f"roi_result_not_found: {roi_result}"})
            continue

        gt_names = {f"{i:05d}.png" for i in _list_frame_ids(gt_dir)}
        base_gt_names = {f"{i:05d}.png" for i in _list_frame_ids(baseline_gt)}
        if gt_names != base_gt_names:
            skipped.append({"dataset": dataset, "reason": "gt_filename_mismatch_between_roots"})
            continue

        rows.append(
            _evaluate_method(
                dataset=dataset,
                method="baseline",
                gt_dir=gt_dir,
                result_dir=base_result,
                fps=args.fps,
                roi_threshold=args.roi_threshold,
                basnet=basnet,
                lpips_metric=lpips_metric,
                device=device,
                max_frames=max_frames,
            )
        )
        rows.append(
            _evaluate_method(
                dataset=dataset,
                method="roi_promptus",
                gt_dir=gt_dir,
                result_dir=roi_result,
                fps=args.fps,
                roi_threshold=args.roi_threshold,
                basnet=basnet,
                lpips_metric=lpips_metric,
                device=device,
                max_frames=max_frames,
            )
        )

    per_dataset_rows = [_to_row(r) for r in rows]
    overall_rows = _build_overall(rows)

    _write_csv(out_dir / "per_dataset_metrics.csv", per_dataset_rows)
    (out_dir / "per_dataset_metrics.json").write_text(
        json.dumps({"rows": per_dataset_rows, "skipped": skipped}, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "overall_comparison.csv", overall_rows)
    _write_md(out_dir / "overall_comparison.md", overall_rows)

    print(f"Done. Wrote files to: {out_dir}")
    print(f"Datasets processed: {sorted({r.dataset for r in rows})}")
    if skipped:
        print("Skipped datasets:")
        for item in skipped:
            print(f"  - {item['dataset']}: {item['reason']}")


if __name__ == "__main__":
    main()

