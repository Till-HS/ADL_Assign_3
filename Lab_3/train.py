import json
import os

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.transforms import transforms

from datasets import CaptionDataset
from models import RNN, ImageEncoder
from utils import save_checkpoint, clip_gradient, accuracy, AverageMeter

# data_parameters
data_folder = './data/processed/'
data_name = 'coco_5_cap_per_img_5_min_word_freq'

# model_parameters
rnn_dim = 256
attention_dim = 256
embed_dim = 256
encoding_pixel_len = 14
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training_parameters
start_epoch = 0
epochs = 100
epochs_since_improvement = 0
print_freq = 100
rnn_lr = 4e-4
batch_size = 32
workers = 1
grad_clip = 5.
best_validation = float('inf')
checkpoint = None  # path to checkpoint, None if none

def train(train_loader, encoder, rnn, criterion, rnn_optimizer, epoch):
    # activate training
    rnn.train()
    encoder.train()

    # initialize metrics
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        # to GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # forward
        imgs = encoder.forward(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = rnn.forward(imgs, caps, caplens)

        # remove <start> from target since it was not predicted but given
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Back prop.
        rnn_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(rnn_optimizer, grad_clip)

        # Update weights
        rnn_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          top5=top5accs))

def validate(val_loader, encoder, rnn, criterion):
    # eval mode
    rnn.eval()
    encoder.eval()

    # initialize metrics
    losses = AverageMeter()
    top5accs = AverageMeter()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            # to GPU
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # forward
            imgs = encoder.forward(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = rnn.forward(imgs, caps, caplens)

            # remove <start> from target since it was not predicted but given
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(scores, targets)

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            # Print status
            if i % print_freq == 0:
                print('validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(val_loader),
                                                                              loss=losses,
                                                                              top5=top5accs))
    return losses.avg


if __name__ == '__main__':
    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # initialize models and optimizers
    if checkpoint is None:
        rnn = RNN(attention_dim=attention_dim,
                      embed_dim=embed_dim,
                      rnn_dim=rnn_dim,
                      vocab_size=len(word_map))
        rnn_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, rnn.parameters()),
                                             lr=rnn_lr)
        encoder = ImageEncoder(encoding_pixel_len = 14)
    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_validation = checkpoint['validation']
        rnn = checkpoint['rnn']
        rnn_optimizer = checkpoint['rnn_optimizer']
        encoder = checkpoint['encoder']
    encoder.to(device)
    rnn.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # load data
    transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL', transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):
        # stop if no improvement has been made
        if epochs_since_improvement == 20:
            break

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              rnn=rnn,
              criterion=criterion,
              rnn_optimizer=rnn_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_validation = validate(val_loader=val_loader,
                                     encoder=encoder,
                                     rnn=rnn,
                                     criterion=criterion)

        # Check if there was an improvement
        is_best = recent_validation < best_validation
        best_validation = min(recent_validation, best_validation)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, rnn,
                        rnn_optimizer, recent_validation, is_best)


