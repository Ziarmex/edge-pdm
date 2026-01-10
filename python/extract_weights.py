import json, tensorflow as tf, numpy as np

interp = tf.lite.Interpreter(model_path="anomaly_model.tflite")
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]["index"], np.zeros((1, 64), dtype=np.float32))
interp.invoke()

data = {}
for d in interp.get_tensor_details():
    try:
        t = interp.get_tensor(d["index"])
        if t is not None and len(t.shape) > 0:
            data[d["name"]] = t.tolist()
    except:
        pass

with open("model_weights.json", "w") as f:
    json.dump(data, f)

matmuls = [k for k in data if "MatMul" in k and ";" not in k]
biases = [k for k in data if "BiasAdd" in k and ";" not in k]
print(f"✓ {len(matmuls)} poids + {len(biases)} biais -> model_weights.json ({len(json.dumps(data))} bytes)")
