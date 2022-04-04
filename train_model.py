import torch
import os

import wandb

# define step counter globally
step = 0


def train(model, optim, scheduler, criterion, train_set, device, epochs=2, grad_clip_value=1.0, print_interval=100,
          model_save_path="./ckgcnn.pt", save_interval=100, test_fn=None, global_stepcount=True):
    """

    :param model:
    :param optim:
    :param criterion:
    :param train_set:
    :param device:
    :param epochs:
    :param grad_clip_value:
    :param print_interval:
    :param model_save_path:
    :param save_interval:
    :param test_fn:
    :param global_stepcount:
    """

    total_samples = 0
    if global_stepcount:
        global step
    else:
        step = 0

    best_acc = 0.

    for epoch in range(epochs):
        model.train()

        # Accumulate accuracy and loss
        running_loss = 0
        running_corrects = 0

        for iteration, (samples, labels) in enumerate(train_set):
            optim.zero_grad()

            samples = samples.to(device)
            labels = labels.to(device)

            # forward pass
            out = model(samples)
            loss = criterion(out, labels)

            # backward pass, gradient clipping and weight update
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(), grad_clip_value
            # )
            optim.step()

            # keep track of running loss and correctly classified samples
            running_loss += (loss.item() * labels.size(0))
            corrects = (torch.max(out, 1)[1] == labels).sum().item()

            running_corrects += corrects
            step += 1
            total_samples += labels.size(0)

            # print running loss every n steps
            if not iteration % print_interval:
                if wandb.run:
                    wandb.log({"epoch": epoch, "loss": loss.item(), "batch_accuracy": corrects/labels.size(0)}, step=step)
                print(f"epoch {epoch} - iteration {iteration} - batch loss {loss.item():.2f} - batch accuracy {corrects / labels.size(0):.2f}")

            # save the model on interval
            if not iteration % save_interval:
                if model_save_path:
                    torch.save(model, model_save_path)

        # save the model after each epoch
        if model_save_path:
            torch.save(model, model_save_path)

        if test_fn:
            val_acc = test_fn()

            if val_acc > best_acc:
                best_acc = val_acc

                if wandb.run:
                    wandb.log({"best_accuracy": best_acc})

            # step learning rate
            if scheduler:
                scheduler.step()
