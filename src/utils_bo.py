import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize


# Perform normalization or standardization based on settings
def apply_transformation(train_x, train_y, bounds, settings):
    if settings.get("normalize", False):
        train_x_normalized = normalize(train_x, bounds)
    else:
        train_x_normalized = train_x  # No normalization

    if settings.get("standardize", False):
        train_y_standardized = standardize(train_y)
    else:
        train_y_standardized = train_y  # No standardization

    return train_x_normalized, train_y_standardized


# Generate initial data based on Sobol samples
def generate_initial_data(objective, bounds, n=5, device="cpu", dtype=torch.float):
    initial_x = (
        draw_sobol_samples(bounds=bounds, n=n, q=1)
        .squeeze(1)
        .to(device=device, dtype=dtype)
    )
    initial_y = objective(initial_x).to(device)
    return initial_x, initial_y
