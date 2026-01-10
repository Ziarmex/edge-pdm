import json, os, numpy as np

_weights = None

def _load():
    global _weights
    if _weights is not None:
        return
    path = os.path.join(os.path.dirname(__file__), "model_weights.json")
    with open(path) as f:
        raw = json.load(f)
    _weights = {
        "dense_1":   (np.array(raw["functional_1/dense_1/MatMul"]),
                       np.array(raw["functional_1/dense_1/Relu;functional_1/dense_1/BiasAdd"])),
        "dense_1_2": (np.array(raw["functional_1/dense_1_2/MatMul"]),
                       np.array(raw["functional_1/dense_1_2/Relu;functional_1/dense_1_2/BiasAdd"])),
        "dense_2_1": (np.array(raw["functional_1/dense_2_1/MatMul"]),
                       np.array(raw["functional_1/dense_2_1/Relu;functional_1/dense_2_1/BiasAdd"])),
        "dense_3_1": (np.array(raw["functional_1/dense_3_1/MatMul"]),
                       np.array(raw["functional_1/dense_3_1/Relu;functional_1/dense_3_1/BiasAdd"])),
        "dense_4_1": (np.array(raw["functional_1/dense_4_1/MatMul"]),
                       np.array(raw["functional_1/dense_4_1/Relu;functional_1/dense_4_1/BiasAdd"])),
        "dense_5_1": (np.array(raw["functional_1/dense_5_1/MatMul"]),
                       np.array(raw["functional_1/dense_5_1/BiasAdd"])),
    }

def relu(x):
    return np.maximum(x, 0)

def predict(x):
    """x: (64,) numpy array. Returns reconstructed (64,) numpy array."""
    _load()
    h = x
    # Encoder
    h = relu(h @ _weights["dense_1"][0].T + _weights["dense_1"][1])
    h = relu(h @ _weights["dense_1_2"][0].T + _weights["dense_1_2"][1])
    h = relu(h @ _weights["dense_2_1"][0].T + _weights["dense_2_1"][1])
    # Decoder
    h = relu(h @ _weights["dense_3_1"][0].T + _weights["dense_3_1"][1])
    h = relu(h @ _weights["dense_4_1"][0].T + _weights["dense_4_1"][1])
    h = h @ _weights["dense_5_1"][0].T + _weights["dense_5_1"][1]
    return h

def mse(x, y):
    return float(np.mean(np.square(x - y)))
