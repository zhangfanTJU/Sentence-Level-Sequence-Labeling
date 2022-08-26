import torch.nn as nn
from module.layers import LSTM, drop_sequence_sharedmask


class Encoder(nn.Module):
    def __init__(self, input_size, config, inputs='word'):
        super(Encoder, self).__init__()
        self.dropout = config.dropout_mlp

        if inputs == 'word':
            hidden_size = config.word_hidden_size
            num_layers = config.word_num_layers
        elif inputs == 'sent':
            hidden_size = config.sent_hidden_size
            num_layers = config.sent_num_layers

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in=config.dropout_input,
            dropout_out=config.dropout_hidden,
        )

    def forward(self, inputs, masks):
        # input:  b x len x input_size
        # masks: b x len

        hiddens, _ = self.lstm(inputs, masks)  # len x b x hidden*2
        hiddens.transpose_(1, 0)  # b x len x hidden*2

        if self.training:
            hiddens = drop_sequence_sharedmask(hiddens, self.dropout)

        return hiddens
