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
    'encoder_bidirectional': False  # Whether the encoder RNN is bidirectional
}

model = Seq2SeqModel(num_classes=28, **kwargs)

# Get number of model parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")