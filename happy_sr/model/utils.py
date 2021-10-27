import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

chars = '\', ,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(
    ',')


def text_to_int(text):
    return [chars.index(char) for char in text]


def int_to_text(labels):
    return ''.join([chars[int(label)] for label in labels])


def get_optimizer(model, loader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-6, steps_per_epoch=int(len(loader)), epochs=epochs, anneal_strategy='linear')

    return optimizer, scheduler


def get_loss():
    loss = torch.nn.CTCLoss(blank=28).to(device)

    return loss


def levenshtein_distance(a, b):
    distances = np.zeros((len(a)+1, len(b)+1))

    for i in range(len(a)+1):
        distances[i][0] = i

    for i in range(len(b)+1):
        distances[0][i] = i

    if i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i] == b[j]:
                distances[i][j] = distances[i-1][j-1]

            else:
                distances[i][j] = min([
                    distances[i][j-1],
                    distances[i-1][j],
                    distances[i-1][j-1],
                ]) + 1

    return distances[len(a)][len(b)]


def greedy_decoder(output, labels, blank_label=28, collapse_repeated=True):
    decoded = []
    targets = [[l.item() for l in label] for label in labels]

    for phrase in torch.argmax(output, dim=2):
        decoded.append([0])
        previous = ''

        for arg in phrase:
            if arg != blank_label:
                if not(collapse_repeated and previous == arg):
                    decoded[-1].append(arg.item())
                    previous = arg.item()

            else:
                previous = ''

    decoded, labels = [int_to_text(d) for d in decoded], [
        int_to_text(l) for l in targets]

    return decoded, labels


def error_rate(output, labels):
    rates = []

    for a, b in zip(output, labels):
        rates.append(levenshtein_distance(a, b) / len(a))

    return rates


def avg(array):
  return sum(array)/len(array)


def word_error_rate(output, labels):
    output, labels = greedy_decoder(output, labels)
    return avg(error_rate([o.split(' ') for o in output], [l.split(' ') for l in labels]))


def char_error_rate(output, labels):
    return avg(error_rate(*greedy_decoder(output, labels)))
