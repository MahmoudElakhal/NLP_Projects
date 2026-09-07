import torch.nn as nn


class ToxicLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes ):

        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
            if num_layers > 1
            else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim * 2, num_classes )

    def forward(self, x):

        # [batch, sequence]
        x = self.embedding(x)

        # [batch, sequence, embedding]
        _, (hidden, _) = self.lstm(x)

        # Last forward hidden state
        forward_hidden = hidden[-2]

        # Last backward hidden state
        backward_hidden = hidden[-1]

        # Concatenate
        hidden_state = __import__(
            "torch"
        ).cat(
            (
                forward_hidden,
                backward_hidden
            ),
            dim=1
        )

        hidden_state = self.dropout(
            hidden_state
        )

        # Raw logits
        logits = self.fc(
            hidden_state
        )

        return logits