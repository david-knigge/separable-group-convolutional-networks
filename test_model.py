import torch
import wandb


def test(model, test_set, device, loss=None, limit=None):
    """
    Evaluate the classification accuracy of a given model on a given test set.

    :param model:
    :param test_set:
    :return:
    """
    correct = 0
    total = 0
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        model.eval()
        for data in test_set:
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            total += labels.size(0)
            correct += (torch.max(out, 1)[1] == labels).sum().item()
            total_batches += 1

            if loss:
                total_loss += loss(out, labels).item()

            if limit:
                if total > limit:
                    break

    print(f"test set accuracy on {total} samples: {(100 * correct / total)}")

    if wandb.run:
        wandb.log({"test_accuracy": correct/total, "test_loss": total_loss/total_batches})
    return correct / total

