import torch
from torchviz import make_dot
from modeling import Seq2SeqModel


# You can add or modify your Seq2SeqModel's hyperparameter (keys and values)
kwargs = {
    'hidden_dim': 128,   # Hidden dimension size for RNN
    'n_rnn_layers': 3,  # Number of RNN layers
    'rnn_dropout': 0.3, # Dropout rate for RNN
    'rnn_type': 'GRU', # Type of RNN ('LSTM' or 'GRU')
    'cnn_settings': {   # Settings for the CustomCNN
        'block1_dim': 32,
        'block2_dim': 64,
        'block3_dim': 128,
        'fc_dim': 128,
        'model_type': 'VGG'  # Type of CNN ('VGG' or 'ResNet')
    },
    'encoder_bidirectional': True  # Whether the encoder RNN is bidirectional
}

model = Seq2SeqModel(num_classes=28, **kwargs)

# Create dummy inputs
dummy_input = torch.randn(2, 8, 28, 28, 3)  # (Batch_size, Sequence_length, Height, Width, Channel)
dummy_lengths = torch.tensor([8, 8])  # Lengths of sequences in the batch
dummy_inp_seq = torch.randint(0, 28, (2, 8))  # (Batch_size, Sequence_length)

# Forward pass through the model
outputs, hidden_state = model(dummy_input, dummy_lengths, dummy_inp_seq)

# Create a visualization graph
dot = make_dot(outputs, params=dict(model.named_parameters()))

# Save the graph to a file
dot.format = 'png'
dot.render("seq2seq_model_graph")