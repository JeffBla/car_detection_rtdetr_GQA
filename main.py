import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from ds import GTACarDataset, collate_fn
from model import RTDetrGQAForObjectDetection


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
def evaluate(detector, data_loader, device):
    metric = MeanAveragePrecision(iou_type="bbox",
                                  iou_thresholds=[0.85],
                                  class_metrics=True)
    detector.eval()

    for batch in data_loader:
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
        targets = _prepare_targets(batch["labels"])
        metric.update(preds, targets)

    return metric.compute()

def train(detector, config):
    device = config["device"]
    detector = detector.to(device)

    train_dataset = GTACarDataset(config["root_dir"], "train",
                                  detector.processor)
    val_dataset = GTACarDataset(config["root_dir"], "val", detector.processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = AdamW(detector.parameters(),
                      lr=config["lr"],
                      weight_decay=config["weight_decay"])
    num_epochs = config["num_epochs"]

    best_score = 0.0
    for epoch in range(num_epochs):
        detector.train()
        total_loss = 0.0

        for batch in train_loader:
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

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}: train loss = {total_loss / len(train_loader):.4f}"
        )

        if (epoch + 1) % config["eval_interval"] == 0:
            metrics = evaluate(detector, val_loader, device)
            current_map = metrics["map"].item()
            print(f"Validation mAP@0.85: {current_map:.4f}")

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


if __name__ == "__main__":
    config = None
    model = RTDetrGQAForObjectDetection(config)

    if config["train"]:
        train(model, config)
