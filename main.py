import argparse
import json
from pathlib import Path
import torch
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

from ds import GTACarDataset, collate_fn
from model import RTDetrGQAForObjectDetection


def num_parameters(model):
    return sum(p.numel() for p in model.parameters())


def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = boxes.unbind(-1)
    b = [
        x_c - 0.5 * w,
        y_c - 0.5 * h,
        x_c + 0.5 * w,
        y_c + 0.5 * h,
    ]
    return torch.stack(b, dim=-1)


def _prepare_targets(batch_labels):
    targets = []
    for label in batch_labels:
        boxes = label["boxes"]
        size = label["orig_size"]
        if not isinstance(size, torch.Tensor):
            size = torch.tensor(size, dtype=boxes.dtype, device=boxes.device)
        else:
            size = size.to(device=boxes.device)
        size = size.float()
        h, w = size.unbind(-1)
        scale = torch.tensor([w, h, w, h],
                             dtype=boxes.dtype,
                             device=boxes.device)
        boxes = boxes * scale
        boxes = _box_cxcywh_to_xyxy(boxes)
        targets.append({
            "boxes": boxes.cpu(),
            "labels": label["class_labels"].cpu(),
        })
    return targets


@torch.no_grad()
def evaluate(detector, data_loader, device, collect_predictions: bool = False):
    metric = MeanAveragePrecision(iou_type="bbox",
                                  iou_thresholds=[0.85],
                                  class_metrics=True)
    detector.eval()
    detections = [] if collect_predictions else None

    for batch in tqdm(data_loader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)

        outputs = detector(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([
            lbl["orig_size"] if isinstance(lbl["orig_size"], torch.Tensor) else
            torch.tensor(lbl["orig_size"]) for lbl in batch["labels"]
        ]).to(device)

        processed_outputs = detector.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes)
        preds = [{
            "boxes": item["boxes"].cpu(),
            "scores": item["scores"].cpu(),
            "labels": item["labels"].cpu(),
        } for item in processed_outputs]
        if collect_predictions:
            image_ids = batch.get("image_ids")
            if image_ids is None:
                raise ValueError("image_ids missing from batch data")
            for img_id, item in zip(image_ids, processed_outputs):
                boxes = item["boxes"].cpu().tolist()
                scores = item["scores"].cpu().tolist()
                labels = item["labels"].cpu().tolist()
                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box
                    bbox = [
                        x_min,
                        y_min,
                        max(0.0, x_max - x_min),
                        max(0.0, y_max - y_min),
                    ]
                    detections.append({
                        "image_id": int(img_id),
                        "category_id": int(label),
                        "bbox": bbox,
                        "score": float(score),
                    })
        targets = _prepare_targets(batch["labels"])
        metric.update(preds, targets)

    return metric.compute(), detections


def parse_args():
    """Parse CLI options for training/evaluation."""
    parser = argparse.ArgumentParser(
        description="RT-DETR training script for GTA car detection.")
    parser.add_argument("--root_dir",
                        type=str,
                        default="./hw3_dataset",
                        help="Path to dataset root directory.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=2,
                        help="Batch size for training and validation loader.")
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4,
                        help="Learning rate for AdamW.")
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-4,
                        help="Weight decay for AdamW.")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=10,
                        help="Number of training epochs.")
    parser.add_argument("--eval_interval",
                        type=int,
                        default=1,
                        help="How often (in epochs) to run validation.")
    parser.add_argument("--checkpoint_path",
                        type=str,
                        help="Pattern for checkpoint path. Empty to disable.")
    parser.add_argument("--num_kv_heads",
                        type=int,
                        default=4,
                        help="KV head count for GroupedQueryAttention.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device identifier, e.g. cuda or cpu.")
    parser.add_argument("--skip_train",
                        action="store_true",
                        help="Skip running training loop (e.g. dry run).")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="Number of optimizer steps to linearly warm up LR.")
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Alternative warmup duration expressed in epochs.")
    parser.add_argument("--log_dir",
                        type=str,
                        default="runs/",
                        help="TensorBoard log directory (default runs/*).")
    parser.add_argument("--hidden_dim_GQA",
                        type=int,
                        default=None,
                        help="Hidden dimension for GQA modules.")
    parser.add_argument(
        "--detections_output",
        type=str,
        default=None,
        help=("When running --skip_train with evaluate_only, write model "
              "predictions (COCO JSON) to this path."),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Dataset split to use for evaluation when skipping training.",
    )
    return parser.parse_args()


def train(detector, config):
    device = config["device"]
    detector = detector.to(device)
    writer = SummaryWriter(log_dir=config.get("log_dir") or None)
    config_snapshot = json.dumps({
        k: str(v)
        for k, v in config.items()
    },
                                 indent=2,
                                 sort_keys=True)
    writer.add_text("config", f"<pre>{config_snapshot}</pre>", global_step=0)

    train_dataset = GTACarDataset(config["root_dir"], "train",
                                  detector.processor)
    val_dataset = GTACarDataset(config["root_dir"], "val", detector.processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=partial(collate_fn, processor=detector.processor),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, processor=detector.processor),
    )

    optimizer = AdamW(detector.parameters(),
                      lr=config["lr"],
                      weight_decay=config["weight_decay"])
    num_epochs = config["num_epochs"]
    steps_per_epoch = len(train_loader)
    total_updates = steps_per_epoch * num_epochs
    warmup_steps = max(
        int(config.get("warmup_steps", 0)),
        int(config.get("warmup_epochs", 0) * steps_per_epoch),
    )
    scheduler = None
    if warmup_steps > 0 and total_updates > 0:
        warmup_steps = min(warmup_steps, total_updates)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            return 1.0

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_score = 0.0
    global_step = 0
    for epoch in tqdm(range(num_epochs)):
        detector.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{
                k: v.to(device)
                for k, v in t.items()
            } for t in batch["labels"]]

            outputs = detector(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,  # RT-DETR 會自己算所有 loss（Hungarian + bbox + cls）
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            writer.add_scalar("train/total_loss", loss.item(), global_step)
            loss_dict = getattr(outputs, "loss_dict", None)
            if loss_dict:
                for name, value in loss_dict.items():
                    writer.add_scalar(f"train/{name}", value.item(),
                                      global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, global_step)
            global_step += 1

        print(
            f"Epoch {epoch+1}: train loss = {total_loss / len(train_loader):.4f}"
        )

        if (epoch + 1) % config["eval_interval"] == 0:
            metrics, _ = evaluate(detector, val_loader, device)
            current_map = metrics["map"].item()
            print(f"Validation mAP@0.85: {current_map:.4f}")
            writer.add_scalar("val/mAP_0.85", current_map, epoch + 1)

            if current_map > best_score:
                best_score = current_map
                ckpt_pattern = config.get("checkpoint_path")
                if ckpt_pattern:
                    torch.save(detector.state_dict(),
                               ckpt_pattern.format(epoch=epoch + 1))
                    print("New best model saved.")
                else:
                    print("New best score (no checkpoint path configured).")

            detector.train()
    writer.flush()
    writer.close()


def evaluate_only(detector, config):
    device = config["device"]
    if config.get("checkpoint_path"):
        checkpoint_path = config["checkpoint_path"]
        detector.load_state_dict(
            torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model checkpoint from {checkpoint_path}")

    model_size = num_parameters(detector)
    print(f"Model size: {model_size}")
    detector = detector.to(device)

    val_dataset = GTACarDataset(config["root_dir"], config.get("eval_split"),
                                detector.processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, processor=detector.processor),
    )

    collect_predictions = bool(config.get("detections_output"))
    metrics, detections = evaluate(detector,
                                   val_loader,
                                   device,
                                   collect_predictions=collect_predictions)
    current_map = metrics["map"].item()
    print(f"Validation mAP@0.85: {current_map:.4f}")
    if collect_predictions and detections is not None:
        output_path = Path(config["detections_output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(detections, indent=2))
        print(f"Saved {len(detections)} detections to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)

    model = RTDetrGQAForObjectDetection(config)

    if not config["skip_train"]:
        train(model, config)
    else:
        evaluate_only(model, config)
