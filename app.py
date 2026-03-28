from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__)

import os

@app.route('/images/<filename>')
def get_image(filename):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(base_dir, filename)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Load Pipeline + Features freshly for each request to prevent state mutation
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline = joblib.load(os.path.join(base_dir, "notebook", "pipeline.pkl"))
        feature_columns = joblib.load(os.path.join(base_dir, "notebook", "feature_columns.pkl"))
        
        data = request.json
        input_data = {col: 0 for col in feature_columns}
        
        def reset_group(prefix):
            for col in feature_columns:
                if col.startswith(prefix):
                    input_data[col] = 0
                    
        reset_group("insured_education_level_")
        reset_group("insured_sex_")
        reset_group("incident_severity_")
        reset_group("insured_relationship_")
        reset_group("insured_hobbies_")
        
        # Default values for fields missing in UI mockup but required by model
        input_data["age"] = float(data.get("age", 39.0))
        input_data["umbrella_limit"] = float(data.get("umbrella_limit", 0.0))
        
        for key, value in data.items():
            if str(value).strip() == "":
                continue
            if key in input_data:
                try:
                    input_data[key] = float(value)
                except ValueError:
                    val_str = str(value).strip().upper()
                    if val_str in ["YES", "MALE", "Y", "TRUE"]:
                        input_data[key] = 1.0
                    elif val_str in ["NO", "FEMALE", "N", "FALSE", "?", "UNKNOWN"]:
                        input_data[key] = 0.0
                    else:
                        # Map strings pseudo-alphabetically to emulate LabelEncoder
                        # or just pass a hash if we don't know the exact alphabet
                        input_data[key] = float(abs(hash(val_str)) % 5)
            elif type(value) is str:
                combined_key = f"{key}_{value}"
                if combined_key in input_data:
                    input_data[combined_key] = 1.0
                    
        input_df = pd.DataFrame([input_data])
        # Add missing columns with 0 if any
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
                
        input_df = input_df[feature_columns]
        
        prediction = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]
        
        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "confidence": float(prob)
        })
    except Exception as e:
        import traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
            f.write("\nData payload:\n" + str(request.json))
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)