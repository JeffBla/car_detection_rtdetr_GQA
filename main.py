import torch
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW

from ds import GTACarDataset, collate_fn
from model import RTDetrGQAForObjectDetection


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
            detector.eval()

            # 在這裡可以加入 validation code，計算 mAP
            # 省略...
            # 這裡可以存 checkpoint，然後用助教提供的 eval code
            # (IoU threshold = 0.85) 去算 validation mAP
            # torch.save(model.state_dict(), f"ckpt_epoch{epoch+1}.pth")


if __name__ == "__main__":
    config = None
    model = RTDetrGQAForObjectDetection(config)

    if config["train"]:
        train(model, config)
