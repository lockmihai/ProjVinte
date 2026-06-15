"""
model.py
Modele de rețele neuronale: LSTM, GRU pentru regresie și clasificare / Neural Network Models: LSTM, GRU for regression and classification.
"""
import torch
import torch.nn as nn


class LSTMPredictor(nn.Module):
    """
    LSTM pentru regresie / LSTM for regression tasks.

    input shape:  (batch, seq_len, input_size)
    output shape: (batch, output_size)
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super().__init__()

        # Recurrent LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, # Shape is (batch_size, sequence_length, features)
            dropout=dropout if num_layers > 1 else 0 # Only apply dropout if depth is > 1
        )

        # Dropout and fully connected layer to map LSTM hidden state to outputs
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the LSTM layer
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n (hidden state): (num_layers, batch, hidden_size)
        # c_n (cell state): (num_layers, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take the hidden state of the last layer for the last step in the sequence
        last_hidden = h_n[-1]  # (batch, hidden_size)
        
        # Pass through dropout and final linear mapping layer
        out = self.fc(self.dropout(last_hidden))
        return out


class GRUPredictor(nn.Module):
    """
    GRU pentru regresie / GRU for regression tasks.
    GRU is computationally faster than LSTM, and shows similar results on financial series.
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 output_size: int = 1):
        super().__init__()

        # Recurrent GRU layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass through the GRU layer
        # h_n: (num_layers, batch, hidden_size)
        gru_out, h_n = self.gru(x)
        
        # Take the hidden state of the last layer for the last step
        last_hidden = h_n[-1]
        
        # Pass through dropout and final linear layer
        out = self.fc(self.dropout(last_hidden))
        return out


class LSTMClassifier(nn.Module):
    """
    LSTM pentru clasificare binară / LSTM for binary classification (e.g., trend direction prediction: ↑/↓).
    """
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        # Recurrent LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Classification head with dropout, linear layer, ReLU activation, and sigmoid output
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid() # Squash predictions between 0 and 1 (probabilities)
        )

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.classifier(last_hidden)
        return out


def count_parameters(model: nn.Module) -> int:
    """Numără parametrii antrenabili / Count the number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Simple validation tests
    # Test Regression
    x = torch.randn(4, 60, 25)
    lstm = LSTMPredictor(25, 128, 2, 0.2, output_size=6)
    gru = GRUPredictor(25, 128, 2, 0.2, output_size=6)
    print(f"LSTM: {x.shape} → {lstm(x).shape}, params: {count_parameters(lstm):,}")
    print(f"GRU:  {x.shape} → {gru(x).shape}, params: {count_parameters(gru):,}")

    # Test Classification
    clf = LSTMClassifier(25, 128, 2, 0.2)
    y_clf = clf(x)
    print(f"LSTM Classifier: {x.shape} → {y_clf.shape}, params: {count_parameters(clf):,}")
