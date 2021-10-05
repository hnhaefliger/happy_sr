import torch
import torchaudio
import re


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

chars = ' ,<SPACE>,a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z'.split(',')


def text_to_int(text):
    return [chars.index(char) for char in text]


def int_to_text(labels):
    return ''.join([chars[label] for label in labels])

train_audio_transforms = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        win_length=int(32*48000/1000),
        hop_length=int(10*48000/1000),
        n_fft=int(32*48000/1000),
        n_mels=28#n_mels,
    ),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35),
)

valid_audio_transforms = torch.nn.Sequential(
    torchaudio.transforms.MelSpectrogram(
        sample_rate=48000,
        win_length=int(32*48000/1000),
        hop_length=int(10*48000/1000),
        n_fft=int(32*48000/1000),
        n_mels=28#n_mels,
    ),
)


def prepare_training_data(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, dictionary) in data:
        spectrogram = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)

        spectrogram = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)

        spectrograms.append(spectrogram)

        label = torch.Tensor(text_to_int(re.sub('[^a-zA-Z ]+', '', dictionary['sentence'].lower())))
        labels.append(label)

        input_lengths.append(spectrogram.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def prepare_testing_data(data):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for (waveform, _, dictionary) in data:
        spectrogram = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)

        spectrograms.append(spectrogram)

        label = torch.Tensor(text_to_int(re.sub('[^a-zA-Z ]+', '', dictionary['sentence'].lower())))
        labels.append(label)

        input_lengths.append(spectrogram.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths


def get_training_data(n_mels, batch_size=16, root='./cv-valid-train', tsv='train.tsv'):
    train_dataset = torchaudio.datasets.COMMONVOICE(root, tsv)

    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=prepare_training_data,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_testing_data(n_mels, batch_size=16, root='./cv-valid-test', tsv='test.tsv'):
    test_dataset = torchaudio.datasets.COMMONVOICE(root, tsv)

    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=prepare_testing_data,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
