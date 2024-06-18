import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.init import kaiming_normal_, constant_


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
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

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
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

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
                kaiming_normal_(m.weight)
                if m.bias is not None:
                    constant_(m.bias, 0)

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


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, nhead=8, dropout=0.1, cnn_settings=None):
        super(TransformerEncoder, self).__init__()
        self.cnn = CustomCNN(**cnn_settings) if cnn_settings else CustomCNN()
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

    def forward(self, inputs, lengths):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        output: (Batch_size, Sequence_length, Hidden_dim)
        """
        x = self.cnn(inputs)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        return output


# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab=28, hidden_dim=64, num_layers=3, nhead=8, dropout=0.5):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(n_vocab, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layers, num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, n_vocab)

    def forward(self, input_seq, memory):
        embedded = self.embedding(input_seq)
        embedded = self.pos_encoder(embedded)
        output = self.transformer_decoder(embedded, memory)
        output = self.fc_out(output)
        return output


# Full Seq2Seq model
class Seq2SeqTransformerModel(nn.Module):
    def __init__(
        self,
        num_classes=28,
        hidden_dim=128,
        nhead=8,
        num_layers=3,
        cnn_settings=None,
        decoder_dropout=0.5,
    ):
        super(Seq2SeqTransformerModel, self).__init__()
        self.encoder = TransformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=decoder_dropout,
            cnn_settings=cnn_settings,
        )
        self.decoder = TransformerDecoder(
            n_vocab=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            dropout=decoder_dropout,
        )

    def forward(self, inputs, lengths, inp_seq):
        memory = self.encoder(inputs, lengths)
        logits = self.decoder(inp_seq, memory)
        return logits

    def generate(self, inputs, lengths, inp_seq, max_length):
        memory = self.encoder(inputs, lengths)
        generated_tok = inp_seq
        outputs = []
        for _ in range(max_length):
            logits = self.decoder(generated_tok, memory)
            _, next_token = torch.max(logits[:, -1, :], dim=1)
            outputs.append(next_token.unsqueeze(1))
            generated_tok = torch.cat([generated_tok, next_token.unsqueeze(1)], dim=1)
        generated_tok = torch.cat(outputs, dim=1)
        return generated_tok
