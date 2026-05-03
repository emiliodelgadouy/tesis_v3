from dataclasses import dataclass

from .imports import copy, torch, tqdm


@dataclass
class TrainConfig:
    epochs: int = 5


def _run_epoch(model, loader, criterion, optimizer, device, training: bool):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def fit(model, train_loader, val_loader, criterion, optimizer, device, config: TrainConfig):
    model = model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_metrics = _run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
            }
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

    model.load_state_dict(best_state)
    return {"model": model, "history": history, "best_val_acc": best_val_acc}
