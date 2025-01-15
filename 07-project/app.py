from flask import Flask, jsonify, request
import wandb
import joblib
import numpy as np

# Log in to Weights & Biases
wandb.login(key="18749c89b0fc53ef93f6d3a4054c5b45067b42cf")

# Initialize a W&B run
wandb.init(
    project="07-project",  
    name="experiment_6",          
)

# loading the model artifact
artifact = wandb.use_artifact("RFC:v2", type="model")
model_dir = artifact.download() 
model_path = f"{model_dir}/random_forest_model.joblib"  # Adjust to the actual model filename 

# loading the model using joblib and assign it to the model variable
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})  # Extract single prediction from array

if __name__ == '__main__':
    app.run( host="0.0.0.0", port=5000, debug=True)
    wandb.finish()
