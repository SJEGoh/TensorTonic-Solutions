import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    h_prev = h_0
    hidden = []
    batch_size, T, input_dim = X.shape

    for t in range(0, T):
        x_t = X[:, t,:]
        h_t = np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)
        hidden.append(h_t)
        h_prev = h_t
    
    return (np.stack(hidden, axis = 1), h_t)
    # YOUR CODE HERE
    pass