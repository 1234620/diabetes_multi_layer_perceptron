from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import os
import io
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
import tempfile

# tensorflow is heavy; import lazily when needed
from tensorflow import keras

app = FastAPI(title="Diabetes MLP Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    age: float = Field(..., description="Age in years")
    gender: str = Field(..., description="M or F")
    bmi: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    creatinine: float
    bun: float


class InferenceState:
    def __init__(self) -> None:
        self.model: Optional[keras.Model] = None
        self.meta: Dict[str, Any] = {}
        self.feature_columns: Optional[List[str]] = None
        self.scaler_mean_: Optional[List[float]] = None
        self.scaler_scale_: Optional[List[float]] = None

    def reset(self) -> None:
        self.model = None
        self.meta = {}
        self.feature_columns = None
        self.scaler_mean_ = None
        self.scaler_scale_ = None

    def load(self, model_bytes: bytes, meta_bytes: bytes, classes_bytes: bytes, scaler_bytes: bytes) -> None:
        # Load model from a temporary file to satisfy load_model's path expectations
        tmp_model_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(model_bytes)
                tmp_model_path = tmp.name
            self.model = keras.models.load_model(tmp_model_path)
        finally:
            if tmp_model_path and os.path.exists(tmp_model_path):
                try:
                    os.remove(tmp_model_path)
                except OSError:
                    pass

        # Parse meta
        meta = json.loads(meta_bytes.decode("utf-8"))
        self.meta = meta

        # Parse classes from separate file
        try:
            classes_data = json.loads(classes_bytes.decode("utf-8"))
            # Handle different possible formats:
            # 1. Direct list: ["Negative", "Positive"]
            # 2. Object with classes_ key: {"classes_": ["Negative", "Positive"]}
            # 3. Object with class_indices (from training code): {"class_indices": {"Negative": 0, "Positive": 1}}
            if isinstance(classes_data, list):
                self.meta["classes_"] = classes_data
            elif isinstance(classes_data, dict):
                if "classes_" in classes_data:
                    self.meta["classes_"] = classes_data["classes_"]
                elif "class_indices" in classes_data:
                    # Convert class_indices dict to sorted list by index
                    # e.g., {"Negative": 0, "Positive": 1} -> ["Negative", "Positive"]
                    class_indices = classes_data["class_indices"]
                    if isinstance(class_indices, dict):
                        # Sort by index and extract class names
                        sorted_classes = sorted(class_indices.items(), key=lambda x: x[1])
                        self.meta["classes_"] = [cls_name for cls_name, idx in sorted_classes]
                    else:
                        self.meta["classes_"] = class_indices
                else:
                    # Try to find any list value that might be classes
                    for key, value in classes_data.items():
                        if isinstance(value, list):
                            self.meta["classes_"] = value
                            break
        except Exception as e:
            # If classes file parsing fails, we'll use defaults later
            # Log error for debugging (can remove in production)
            import traceback
            traceback.print_exc()
            pass

        # Parse scaler from separate file
        scaler_feature_columns = None
        try:
            scaler_data = json.loads(scaler_bytes.decode("utf-8"))
            # Handle format: {"mean": [...], "scale": [...], "feature_columns": [...]}
            if isinstance(scaler_data, dict):
                if "mean" in scaler_data and "scale" in scaler_data:
                    self.scaler_mean_ = scaler_data["mean"]
                    self.scaler_scale_ = scaler_data["scale"]
                    # Store feature_columns from scaler (these match the scaler's mean/scale order)
                    if "feature_columns" in scaler_data:
                        scaler_feature_columns = scaler_data["feature_columns"]
        except Exception as e:
            # If scaler file parsing fails, we'll use identity transform
            import traceback
            traceback.print_exc()
            pass

        # expected keys (best case): classes_, feature_columns, scaler_mean_, scaler_scale_
        # Priority: scaler feature_columns > meta feature_columns > default
        # Use scaler feature_columns if available (they match the scaler's mean/scale order)
        if scaler_feature_columns and isinstance(scaler_feature_columns, list):
            self.feature_columns = scaler_feature_columns
        elif not self.feature_columns or not isinstance(self.feature_columns, list):
            self.feature_columns = meta.get("feature_columns")
        
        # Also check meta for scaler (backward compatibility)
        if self.scaler_mean_ is None:
            self.scaler_mean_ = meta.get("scaler_mean_")
        if self.scaler_scale_ is None:
            self.scaler_scale_ = meta.get("scaler_scale_")

        # Determine model expected input dimension
        try:
            expected_dim = int(self.model.input_shape[-1])
        except Exception:
            expected_dim = None

        # Build fallback feature columns if meta is minimal; then pad/truncate to expected_dim
        def default_feature_list() -> List[str]:
            default_numeric = [
                "Age",
                "BMI",
                "Chol",
                "Triglycerides (TG)",
                "High-Density Lipoprotein (HDL)",
                "Low-Density Lipoprotein (LDL)",
                "Creatinine",
                "BUN (Blood Urea Nitrogen)",
            ]
            gender_cols = ["Gender_F", "Gender_M"]
            return default_numeric + gender_cols

        if not self.feature_columns or not isinstance(self.feature_columns, list):
            self.feature_columns = default_feature_list()

        if expected_dim is not None:
            if len(self.feature_columns) < expected_dim:
                # pad with zero-filled placeholders
                placeholders_needed = expected_dim - len(self.feature_columns)
                self.feature_columns = self.feature_columns + [f"_extra_{i+1}" for i in range(placeholders_needed)]
            elif len(self.feature_columns) > expected_dim:
                # truncate to match model
                self.feature_columns = self.feature_columns[:expected_dim]

        # If scaler is missing, we will use identity transform


STATE = InferenceState()


@app.post("/upload")
async def upload(model: UploadFile = File(...), meta: UploadFile = File(...), classes: UploadFile = File(...), scaler: UploadFile = File(...)):
    if not model.filename.endswith(".keras"):
        raise HTTPException(status_code=400, detail="Model must be a .keras file")
    if not meta.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Meta must be a .json file")
    if not classes.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Classes must be a .json file")
    if not scaler.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Scaler must be a .json file")

    model_bytes = await model.read()
    meta_bytes = await meta.read()
    classes_bytes = await classes.read()
    scaler_bytes = await scaler.read()

    try:
        STATE.load(model_bytes, meta_bytes, classes_bytes, scaler_bytes)
    except Exception as e:
        STATE.reset()
        raise HTTPException(status_code=400, detail=f"Failed to load model/meta/classes/scaler: {e}")

    # Check if classes were successfully loaded
    classes_loaded = STATE.meta.get("classes_")
    has_classes = isinstance(classes_loaded, list) and len(classes_loaded) > 0
    
    response: Dict[str, Any] = {
        "status": "ok",
        "feature_columns": STATE.feature_columns,
        "has_scaler": STATE.scaler_mean_ is not None and STATE.scaler_scale_ is not None,
        "has_classes": has_classes,
        "classes": classes_loaded if has_classes else None,  # Include classes in response for debugging
    }
    return response


def build_dataframe(req: PredictRequest) -> pd.DataFrame:
    # Map input to the original training feature names
    raw = {
        "Age": req.age,
        "BMI": req.bmi,
        "Chol": req.chol,
        "Triglycerides (TG)": req.tg,
        "High-Density Lipoprotein (HDL)": req.hdl,
        "Low-Density Lipoprotein (LDL)": req.ldl,
        "Creatinine": req.creatinine,
        "BUN (Blood Urea Nitrogen)": req.bun,
        "Gender": req.gender.upper(),
    }
    df = pd.DataFrame([raw])
    # one-hot for Gender using pandas get_dummies to match training
    df_encoded = pd.get_dummies(df, columns=["Gender"], prefix=["Gender"])  # Gender_F/Gender_M
    # align to feature_columns, filling missing with 0
    aligned = df_encoded.reindex(columns=STATE.feature_columns, fill_value=0)
    return aligned


def apply_scaling(values: np.ndarray) -> np.ndarray:
    if STATE.scaler_mean_ is None or STATE.scaler_scale_ is None:
        return values
    mean = np.asarray(STATE.scaler_mean_, dtype=np.float32)
    scale = np.asarray(STATE.scaler_scale_, dtype=np.float32)
    # Avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)
    return (values - mean) / scale


@app.post("/predict")
async def predict(req: PredictRequest):
    if STATE.model is None or STATE.feature_columns is None:
        raise HTTPException(status_code=400, detail="Model not loaded. Upload model and meta first.")

    try:
        X_df = build_dataframe(req)
        X = X_df.values.astype(np.float32)
        X = apply_scaling(X)

        preds = STATE.model.predict(X, verbose=0)

        # Handle binary vs multi-class
        # For binary classification (0/1), model outputs 2 classes via softmax
        # preds shape: (1, 2) where [0][0] = P(0=no diabetes), [0][1] = P(1=diabetes)
        classes_meta = STATE.meta.get("classes_")
        
        # Default classes for binary classification if missing from meta
        if not isinstance(classes_meta, list) or len(classes_meta) != 2:
            classes_meta = ["No Diabetes", "Diabetes"]  # 0 = No Diabetes, 1 = Diabetes
        
        if preds.ndim == 2:
            if preds.shape[1] == 2:
                # Binary classification: 2 classes (0=no diabetes, 1=diabetes)
                prob_no_diabetes = float(preds[0][0])
                prob_diabetes = float(preds[0][1])
                label = 1 if prob_diabetes >= 0.5 else 0
                predicted_label = classes_meta[label]
                confidence = prob_diabetes if label == 1 else prob_no_diabetes
                return {
                    "prediction": predicted_label,
                    "probability": confidence,
                    "probabilities": {
                        "no_diabetes": prob_no_diabetes,
                        "diabetes": prob_diabetes
                    }
                }
            elif preds.shape[1] == 1:
                # Single output (sigmoid binary)
                prob = float(preds[0][0])
                label = 1 if prob >= 0.5 else 0
                predicted_label = classes_meta[label]
                return {
                    "prediction": predicted_label,
                    "probability": prob,
                }
            else:
                # Multi-class softmax (3+ classes)
                idx = int(np.argmax(preds[0]))
                confidence = float(preds[0][idx])
                if isinstance(classes_meta, list) and len(classes_meta) > idx:
                    predicted_label = classes_meta[idx]
                else:
                    predicted_label = f"Class {idx}"
                return {
                    "prediction": predicted_label,
                    "probability": confidence,
                    "distribution": preds[0].tolist(),
                }
        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


def mount_static(app: FastAPI):
    static_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    static_dir = os.path.abspath(static_dir)
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


mount_static(app)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8020)), reload=True)


