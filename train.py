import torch.nn.functional as F
import time
import copy
import torch 

from loss import dice_loss

def loss_calculation(pred, target, metrics,  bce_weight=0.5):
    
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)

    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1. - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_loss(metrics, epoch_s, phase):
    out = []

    for key in metrics:
        out.append('{}: {:.6f}'.format(key, metrics[key] / epoch_s))

    print('{}:\n{}'.format(phase, '\n'.join(out)))


def train(model, optimizer, scheduler, data_loader, device, n_epochs=25):
    best_model_cp = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(n_epochs):
        print('Epoch {}/{}\n'.format(epoch + 1, n_epochs))

        start_t = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            metrics = {
                'bce': 0.0,
                'dice': 0.0,
                'loss': 0.0 
            }

            epoch_samples = 0

            for inputs, labels in data_loader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_calculation(outputs, labels, metrics)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0)

            print_loss(metrics, epoch_samples, phase)

            e_loss = metrics['loss'] / epoch_samples

            if phase == 'val' and e_loss < best_loss:
                print("Saving best model")
                best_model_cp = copy.deepcopy(model.state_dict())
                best_loss = e_loss

        end_t = time.time() - start_t
        print('Time elapsed: {}s'.format(end_t))

    print('Best validation loss: {:.6}'.format(best_loss))
    model.load_state_dict(best_model_cp)

    return model
            