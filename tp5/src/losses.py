import numpy as np

def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Para evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return bce

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

def loss_function(y_true, y_pred, z_mean, z_log_var):
    # Pérdida de reconstrucción (Binary Cross-Entropy)
    epsilon = 1e-8  # Para evitar log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    reconstruction_loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred), axis=1)
    reconstruction_loss = np.mean(reconstruction_loss)

    # Divergencia KL
    kl_loss = -0.5 * np.sum(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var), axis=1)
    kl_loss = np.mean(kl_loss)

    # Pérdida total
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss