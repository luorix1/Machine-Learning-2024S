import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init


# Custom CNN definition
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.init_weights()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# Feature extractor
class CustomCNN(nn.Module):
    def __init__(
        self, block1_dim=32, block2_dim=64, block3_dim=128, fc_dim=64, model_type="VGG"
    ):
        super(CustomCNN, self).__init__()

        self.model_type = model_type

        if model_type == "VGG":
            self.block1 = VGGBlock(3, block1_dim)
            self.block2 = VGGBlock(block1_dim, block2_dim)
            self.block3 = VGGBlock(block2_dim, block3_dim)
        elif model_type == "ResNet":
            self.block1 = ResNetBlock(3, block1_dim, stride=1)
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.block2 = ResNetBlock(block1_dim, block2_dim, stride=1)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.block3 = ResNetBlock(block2_dim, block3_dim, stride=1)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError("Unknown model type")

        self.init_weights()

        self.fc = nn.Linear(block3_dim * (28 // 8) * (28 // 8), fc_dim)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, inputs):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        outputs: (Batch_size, Sequence_length, Hidden_dim)
        """
        batch_size, seq_len, height, width, channels = inputs.size()
        x = inputs.view(batch_size * seq_len, channels, height, width)

        if self.model_type == "VGG":
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
        elif self.model_type == "ResNet":
            x = self.block1(x)
            x = self.pool1(x)
            x = self.block2(x)
            x = self.pool2(x)
            x = self.block3(x)
            x = self.pool3(x)

        x = x.view(batch_size, seq_len, -1)

        outputs = self.fc(x)
        return outputs


# Encoder
class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=2,
        rnn_type="LSTM",
        bidirectional=False,
        cnn_settings=None,
    ):
        super(Encoder, self).__init__()
        self.cnn = CustomCNN(**cnn_settings) if cnn_settings else CustomCNN()
        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), hidden_dim)

    def forward(self, inputs, lengths):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths (Batch_size)
        output: (Batch_size, Sequence_length, Hidden_dim)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        x = self.cnn(inputs)
        packed_input = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, hidden_state = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output, hidden_state


# Decoder
class Decoder(nn.Module):
    def __init__(
        self,
        n_vocab=28,
        hidden_dim=64,
        num_layers=2,
        rnn_type="LSTM",
        bidirectional=False,
        pad_idx=0,
        dropout=0.5,
    ):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.bidirectional = bidirectional
        self.n_vocab = n_vocab

        self.embedding = nn.Embedding(n_vocab, hidden_dim, padding_idx=pad_idx)
        rnn_class = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = rnn_class(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim * (2 if bidirectional else 1), n_vocab)

    def forward(self, input_seq, hidden_state):
        """
        input_seq: (Batch_size, Sequence_length)
        output: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        embedded = self.embedding(input_seq)
        output, hidden_state = self.rnn(embedded, hidden_state)
        output = self.lm_head(output)
        return output, hidden_state


# Full Seq2Seq model
class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        num_classes=28,
        hidden_dim=64,
        n_rnn_layers=2,
        rnn_dropout=0.5,
        rnn_type="LSTM",
        cnn_settings=None,
        encoder_bidirectional=False,
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_layers=n_rnn_layers,
            rnn_type=rnn_type,
            bidirectional=encoder_bidirectional,
            cnn_settings=cnn_settings,
        )
        self.decoder = Decoder(
            n_vocab=num_classes,
            hidden_dim=hidden_dim,
            # NOTE: If encoder is bidirectional, number of layers should be doubled
            num_layers=n_rnn_layers * 2 if encoder_bidirectional else n_rnn_layers,
            rnn_type=rnn_type,
            dropout=rnn_dropout,
        )

    def forward(self, inputs, lengths, inp_seq):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, Sequence_length)
        logits: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        encoder_outputs, hidden_state = self.encoder(inputs, lengths)
        logits, hidden_state = self.decoder(inp_seq, hidden_state)
        return logits, hidden_state

    def generate(self, inputs, lengths, inp_seq, max_length):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, 1)
        max_length -> a single integer number (ex. 10)
        generated_tok: (Batch_size, max_length) -> long dtype tensor
        """
        encoder_outputs, hidden_state = self.encoder(inputs, lengths)
        generated_seq = []
        input_token = inp_seq

        for _ in range(max_length):
            output, hidden_state = self.decoder(input_token, hidden_state)
            next_token = output.argmax(dim=-1)
            generated_seq.append(next_token)
            input_token = next_token

        generated_seq = torch.cat(generated_seq, dim=1)
        return generated_seq
