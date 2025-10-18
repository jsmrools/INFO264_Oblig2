# -- Load the dataset to screen (corrupted) --
import numpy as np
def load_npz_X_y(path):
    npz = np.load(path, allow_pickle=True)
    keys = list(npz.files)

    def pick(keys_order):
        for k in keys_order:
            if k in keys:
                return k
        return None

    x_key = pick(["X", "images", "data", "X_train", "X_new", "X_test"])
    if x_key is None:
        x_key = next((k for k in keys if isinstance(npz[k], np.ndarray) and npz[k].ndim >= 2 and npz[k].shape[0] > 1), None)
        if x_key is None:
            raise RuntimeError(f"No suitable X in {path}. Keys={keys}")

    X = np.array(npz[x_key])
    X_flat = X.reshape(len(X), -1) if X.ndim >= 3 else X

    y_key = pick(["y", "labels", "target", "y_train", "y_new", "y_test"])
    y = np.array(npz[y_key]).ravel() if y_key is not None else None
    return X_flat.astype(float), y

def show_bad_data_examples(X, indices=[0, 1, 2, 3, 4]):
    import matplotlib.pyplot as plt

    num_examples = len(indices)
    plt.figure(figsize=(10, 2 * num_examples))
    for i, idx in enumerate(indices):
        plt.subplot(num_examples, 1, i + 1)
        plt.imshow(X[idx].reshape(20, 20), cmap='gray', vmin=0, vmax=255)
        plt.title(f'Example {i + 1} (Index: {idx})')
        plt.axis('off')
    plt.tight_layout()
    plt.show()