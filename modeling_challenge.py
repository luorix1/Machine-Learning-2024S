import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.init as init


# Full Sequence-to-Sequence Model w/ Transformer
class Seq2SeqModel(nn.Module):
    def __init__(
        self,
        num_classes=28,
        hidden_dim=256,
        nhead=4,
        dec_layers=3,
        dim_feedforward=1024,
        dropout=0.2,
        enc_layers=3,
        rnn_dropout=0.4,
        max_length=11,
        cnn_settings=None,
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(
            hidden_dim=hidden_dim,
            num_layers=enc_layers,
            dropout=rnn_dropout,
            cnn_settings=cnn_settings,
        )
        self.decoder = TransformerDecoder(
            output_dim=num_classes,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    # Function that generates masks for the transformer decoder
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, inputs, lengths, input_seq, teacher_forcing_ratio=0.6):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        input_seq: (Batch_size, Sequence_length)
        logits: (Batch_size, Sequence_length, num_classes)
        """
        encoder_output, _ = self.encoder(inputs, lengths)
        seq_length = input_seq.size(1)
        batch_size = inputs.size(0)
        logits = torch.zeros(batch_size, seq_length, self.num_classes).to(inputs.device)
        decoder_input = input_seq[:, :1]  # (Batch_size, 1)  = <sos>

        for t in range(seq_length):
            current_tgt_mask = self.generate_square_subsequent_mask(
                decoder_input.size(1)
            ).to(inputs.device)
            current_tgt_key_padding_mask = decoder_input == 0

            output = self.decoder(
                decoder_input,
                encoder_output,
                current_tgt_mask,
                current_tgt_key_padding_mask,
            )
            logits[:, t] = output[
                :, -1, :
            ]  # (Batch_size, 1, num_classes) -> (Batch_size, num_classes)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            next_input = (
                input_seq[:, t + 1 : t + 2]
                if teacher_force and t < seq_length - 1
                else output[:, -1].argmax(1, keepdim=True)
            )
            decoder_input = torch.cat((decoder_input, next_input), dim=1)

        return logits

    def generate(self, inputs, lengths, start_tokens, max_length=10):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        start_tokens: (Batch_size, 1)
        max_length: a single integer number (e.g., 10)
        generated_tokens: (Batch_size, max_length) -> long dtype tensor
        """
        self.eval()
        with torch.no_grad():
            encoder_output, _ = self.encoder(inputs, lengths)
            generated_tokens = start_tokens
            current_sequence = start_tokens

            for _ in range(max_length):
                current_tgt_mask = self.generate_square_subsequent_mask(
                    current_sequence.size(1)
                ).to(inputs.device)
                current_tgt_key_padding_mask = current_sequence == 0
                decoder_output = self.decoder(
                    current_sequence,
                    encoder_output,
                    current_tgt_mask,
                    current_tgt_key_padding_mask,
                )

                next_token = (
                    decoder_output[:, -1:, :].argmax(-1, keepdim=True).squeeze(-1)
                )
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                current_sequence = torch.cat((current_sequence, next_token), dim=1)

        return generated_tokens[:, 1:]


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
        dropout=0.1,
        cnn_settings=None,
    ):
        super(Encoder, self).__init__()
        self.cnn = CustomCNN(**cnn_settings) if cnn_settings else CustomCNN()
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

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
class TransformerDecoder(nn.Module):
    def __init__(
        self, output_dim, hidden_dim, nhead, num_layers, dim_feedforward, dropout=0.1
    ):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def forward(self, target, encoder_output, target_mask, target_key_padding_mask):
        target_embeddings = self.embedding(target) * torch.sqrt(
            torch.tensor(self.hidden_dim, dtype=torch.float32)
        ).to(target.device)
        target_embeddings = self.pos_encoder(target_embeddings)

        if target_key_padding_mask is not None:
            target_key_padding_mask = target_key_padding_mask.to(target_mask.dtype)

        # Transpose dimensions to match the expected input shape of the transformer decoder
        decoder_output = self.transformer_decoder(
            target_embeddings.transpose(
                0, 1
            ),  # [batch_size, target_seq_len, d_model] -> [target_seq_len, batch_size, d_model]
            encoder_output.transpose(0, 1),
            tgt_mask=target_mask,
            tgt_key_padding_mask=target_key_padding_mask,
        )

        # Transpose back to the original dimensions and pass through the final linear layer
        decoder_output = self.fc_out(decoder_output.transpose(0, 1))
        return decoder_output


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # Add a dimension for batch size
        self.register_buffer("pe", pe)

    def forward(self, x, debug=False):
        if debug:
            print(f"Positional Encoding Input Shape: {x.shape}")
        # Expand the positional encoding to match the batch size
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        if debug:
            print(f"Positional Encoding Output Shape: {x.shape}")
        return self.dropout(x)
