import torch
from torch import nn


class RedditNN(nn.Module):
    """
    Main Class that implements the Long-Short-Term-Memory Neural Network (LSTM NN).
    Inherits from the typical PyTorch nn.Module Class.
    """

    def __init__(self, dataset):
        super(RedditNN, self).__init__()

        self.LSTMSize = 128                 # input/output dimension for LSTM layers
        self.embeddingDimensions = 128      # output dimension for embedding layer
        self.numLayers = 3                  # number of (hidden?) LSTM layers

        numUniqueWords = len(dataset.uniqueWords)

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=numUniqueWords,
            embedding_dim=self.embeddingDimensions,
        )

        # LSTM layer
        self.LSTM = nn.LSTM(
            input_size=self.LSTMSize,
            hidden_size=self.LSTMSize,
            num_layers=self.numLayers,
            dropout=0.2,
        )

        # fully-connected layer
        self.fc = nn.Linear(self.LSTMSize, numUniqueWords)

    def forward(self, x, previousState):
        embedded = self.embedding(x)
        output, state = self.LSTM(embedded, previousState)
        logits = self.fc(output)
        return logits, state

    def init_state(self, sequenceLength):
        return (torch.zeros(self.numLayers, sequenceLength, self.LSTMSize),
                torch.zeros(self.numLayers, sequenceLength, self.LSTMSize))
