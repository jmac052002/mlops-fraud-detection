import joblib
import os
import numpy as np
import json

def model_fn(model_dir):
    """Load model from the model directory"""
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    model_path = os.path.join(model_dir, model_files[0])
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type):
    """Parse input data"""
    if content_type == "text/csv":
        import io
        import numpy as np
        data = np.loadtxt(io.StringIO(request_body), delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return data
    raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Make predictions"""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    return np.column_stack([predictions, probabilities])

def output_fn(prediction, accept):
    """Format output"""
    return json.dumps(prediction.tolist()), accept 