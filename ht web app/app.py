from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import json
import logging

# Import model classes
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm # For lgbm.Booster and LGBMClassifier (though we primarily use Booster for prediction here)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Basic Logging Configuration ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s: %(message)s')
logger = logging.getLogger(__name__)
logger.info("--- Logger initialized ---")

app = Flask(__name__)
logger.info("--- Flask app instance created ---")

MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), 'model_output')
TARGET_COLUMN = 'hp_category'

logger.info(f"--- SCRIPT START: __file__ is {__file__} ---")
logger.info(f"--- SCRIPT START: os.path.dirname(__file__) is {os.path.dirname(__file__)} ---")
logger.info(f"--- MODEL_BASE_DIR configured to: {MODEL_BASE_DIR} ---")
logger.info(f"--- Absolute path for MODEL_BASE_DIR: {os.path.abspath(MODEL_BASE_DIR)} ---")
if not os.path.isdir(MODEL_BASE_DIR):
    logger.error(f"--- CRITICAL: MODEL_BASE_DIR does not exist or is not a directory: {os.path.abspath(MODEL_BASE_DIR)} ---")
else:
    logger.info(f"--- SUCCESS: MODEL_BASE_DIR exists: {os.path.abspath(MODEL_BASE_DIR)} ---")
    logger.info(f"--- Contents of MODEL_BASE_DIR: {os.listdir(MODEL_BASE_DIR)} ---")

MODEL_CONFIG = {
    "catboost": {
        "model_subdir": "catboost_model",
        "model_file": "catboost_model.cbm",
        "scaler_file": "scaler.joblib",
        "features_file": "features.json",
        "loader_class": CatBoostClassifier,
        "load_method": "load_model",
        "is_sklearn_joblib": False
    },
    "xgboost": {
        "model_subdir": "xgboost_model",
        "model_file": "xgboost_model.json",
        "scaler_file": "scaler.joblib",
        "features_file": "features.json",
        "loader_class": XGBClassifier,
        "load_method": "load_model",
        "is_sklearn_joblib": False
    },
    "random_forest": {
        "model_subdir": "randomforest_model",
        "model_file": "randomforest_model.joblib",
        "scaler_file": "scaler.joblib",
        "features_file": "features.json",
        "loader_class": RandomForestClassifier,
        "load_method": "joblib",
        "is_sklearn_joblib": True
    },
    "lightgbm": {
        "model_subdir": "lightgbm_model",
        "model_file": "lightgbm_model.txt",
        "scaler_file": "scaler.joblib",
        "features_file": "features.json",
        "loader_class": None,
        "load_method": "lightgbm_booster",
        "is_sklearn_joblib": False
    },
    "gradient_boosting": {
        "model_subdir": "gradientboosting_model",
        "model_file": "gradientboosting_model.joblib",
        "scaler_file": "scaler.joblib",
        "features_file": "features.json",
        "loader_class": GradientBoostingClassifier,
        "load_method": "joblib",
        "is_sklearn_joblib": True
    }
}
logger.info("--- MODEL_CONFIG defined ---")

loaded_models_cache = {}
logger.info("--- loaded_models_cache initialized ---")

def load_model_components(model_id):
    logger.debug(f"load_model_components: called for model_id: {model_id}")
    if model_id in loaded_models_cache:
        logger.info(f"Using cached components for model_id: {model_id}")
        return loaded_models_cache[model_id]

    if model_id not in MODEL_CONFIG:
        logger.error(f"load_model_components: Invalid model_id: {model_id}. Not found in MODEL_CONFIG.")
        raise ValueError(f"Invalid model_id: {model_id}. Not found in MODEL_CONFIG.")

    config = MODEL_CONFIG[model_id]
    model_subdir_path = os.path.join(MODEL_BASE_DIR, config["model_subdir"])
    logger.debug(f"load_model_components: model_subdir_path: {model_subdir_path} (abs: {os.path.abspath(model_subdir_path)})")

    model_path = os.path.join(model_subdir_path, config["model_file"])
    scaler_path = os.path.join(model_subdir_path, config["scaler_file"])
    features_path = os.path.join(model_subdir_path, config["features_file"])
    logger.debug(f"load_model_components: model_path: {model_path} (abs: {os.path.abspath(model_path)})")
    logger.debug(f"load_model_components: scaler_path: {scaler_path} (abs: {os.path.abspath(scaler_path)})")
    logger.debug(f"load_model_components: features_path: {features_path} (abs: {os.path.abspath(features_path)})")

    if not os.path.exists(model_path):
        logger.error(f"load_model_components: Model file not found: {model_path} (abs: {os.path.abspath(model_path)})")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        logger.error(f"load_model_components: Scaler file not found: {scaler_path} (abs: {os.path.abspath(scaler_path)})")
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(features_path):
        logger.error(f"load_model_components: Features file not found: {features_path} (abs: {os.path.abspath(features_path)})")
        raise FileNotFoundError(f"Features file not found: {features_path}")
    
    logger.info(f"load_model_components: All component files exist for model_id: {model_id}. Proceeding to load.")

    model_instance = None
    try:
        if config["load_method"] == "load_model":
            logger.debug(f"load_model_components: Loading {model_id} using 'load_model' method.")
            ModelClass = config["loader_class"]
            model_instance = ModelClass()
            model_instance.load_model(model_path)
        elif config["load_method"] == "joblib":
            logger.debug(f"load_model_components: Loading {model_id} using 'joblib' method.")
            model_instance = joblib.load(model_path)
        elif config["load_method"] == "lightgbm_booster":
            logger.debug(f"load_model_components: Loading {model_id} using 'lightgbm_booster' method.")
            model_instance = lgbm.Booster(model_file=model_path)
        else:
            logger.error(f"load_model_components: Unknown load_method: {config['load_method']} for model_id: {model_id}")
            raise ValueError(f"Unknown load_method: {config['load_method']} for model_id: {model_id}")
        logger.info(f"load_model_components: Model instance for {model_id} loaded successfully.")
    except Exception as e_model_load:
        logger.error(f"load_model_components: FAILED to load model instance for {model_id} from {model_path}. Error: {e_model_load}", exc_info=True)
        raise

    try:
        logger.debug(f"load_model_components: Loading scaler for {model_id} from {scaler_path}.")
        scaler_instance = joblib.load(scaler_path)
        logger.info(f"load_model_components: Scaler for {model_id} loaded successfully.")
    except Exception as e_scaler_load:
        logger.error(f"load_model_components: FAILED to load scaler for {model_id} from {scaler_path}. Error: {e_scaler_load}", exc_info=True)
        raise

    try:
        logger.debug(f"load_model_components: Loading features for {model_id} from {features_path}.")
        with open(features_path, 'r') as f:
            feature_names_list = json.load(f)
        logger.info(f"load_model_components: Features for {model_id} loaded successfully.")
    except Exception as e_features_load:
        logger.error(f"load_model_components: FAILED to load features for {model_id} from {features_path}. Error: {e_features_load}", exc_info=True)
        raise

    logger.info(f"{model_id.capitalize()} model, scaler, and features loaded and processed successfully.")
    
    components = (model_instance, scaler_instance, feature_names_list)
    loaded_models_cache[model_id] = components
    logger.debug(f"load_model_components: Components for {model_id} cached.")
    return components

@app.route('/')
def index():
    logger.info("--- Route / called ---")
    template_path_check = os.path.join(app.root_path, 'templates', 'index.html')
    logger.debug(f"Attempting to render template: index.html")
    logger.debug(f"app.root_path is: {app.root_path} (abs: {os.path.abspath(app.root_path)})")
    logger.debug(f"Full path to template being considered: {template_path_check} (abs: {os.path.abspath(template_path_check)})")
    
    if not os.path.exists(template_path_check):
        logger.error(f"Template file DOES NOT EXIST at: {template_path_check} (abs: {os.path.abspath(template_path_check)})")
        return "Error: Template file 'index.html' not found in 'templates' folder.", 500
    try:
        logger.info("Rendering template index.html")
        return render_template('index.html')
    except Exception as e_render:
        logger.error(f"Error during render_template('index.html'): {e_render}", exc_info=True)
        return f"Error rendering template: {e_render}", 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("--- Route /predict called ---")
    if 'file' not in request.files:
        logger.warning("/predict: 'file' not in request.files")
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("/predict: No file selected for uploading (file.filename is empty)")
        return jsonify({"error": "No file selected for uploading"}), 400

    model_id_from_request = request.form.get('model_id', 'catboost')
    logger.info(f"Received prediction request for model_id: {model_id_from_request}")

    try:
        logger.debug(f"/predict: Attempting to load model components for {model_id_from_request}")
        model, scaler, feature_names = load_model_components(model_id_from_request)
        logger.info(f"/predict: Model components for {model_id_from_request} loaded successfully.")
    except FileNotFoundError as e_fnf:
        logger.error(f"/predict: FileNotFoundError while loading components for model {model_id_from_request}: {e_fnf}", exc_info=True)
        return jsonify({"error": f"Model files missing for '{model_id_from_request}'. Server configuration issue or files not deployed. Details: {str(e_fnf)}"}), 500
    except Exception as e_load:
        logger.error(f"/predict: Error loading components for model {model_id_from_request}: {e_load}", exc_info=True)
        return jsonify({"error": f"Could not load model '{model_id_from_request}'. Server not ready or model files missing. Details: {str(e_load)}"}), 500

    if not model or not scaler or not feature_names:
        logger.error(f"/predict: Model components for '{model_id_from_request}' are None after load_model_components call. This should not happen.")
        return jsonify({"error": f"Model components for '{model_id_from_request}' are not properly loaded."}), 500

    if file and file.filename.endswith('.csv'):
        try:
            logger.debug(f"/predict: Processing CSV file: {file.filename}")
            input_df = pd.read_csv(file.stream)
            original_input_df_for_metrics = input_df.copy()
            logger.debug(f"/predict: CSV read successfully. Shape: {input_df.shape}. Columns: {original_input_df_for_metrics.columns.tolist()}") # <<< ENHANCED LOGGING

            missing_cols = [col for col in feature_names if col not in input_df.columns]
            if missing_cols:
                logger.warning(f"/predict: Missing required columns in CSV: {', '.join(missing_cols)}")
                return jsonify({"error": f"Missing required columns in CSV: {', '.join(missing_cols)}"}), 400
            
            processed_df = input_df[feature_names].copy()
            for col in feature_names:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')

            if processed_df.isnull().any().any():
                nan_feature_cols = processed_df.columns[processed_df.isnull().any()].tolist()
                logger.warning(f"NaN values found in feature columns: {nan_feature_cols}. Applying fillna(0) as default imputation. Ensure this matches training.")
                processed_df.fillna(0, inplace=True)

            logger.debug("/predict: Scaling features...")
            scaled_features = scaler.transform(processed_df)
            logger.debug("/predict: Features scaled. Predicting...")
            
            raw_predictions_from_model = model.predict(scaled_features)
            logger.info(f"/predict: Raw predictions received from model {model_id_from_request}. Shape: {raw_predictions_from_model.shape if hasattr(raw_predictions_from_model, 'shape') else 'N/A'}")
            
            predictions = None

            if model_id_from_request == "lightgbm" and isinstance(model, lgbm.Booster):
                if raw_predictions_from_model.ndim == 2 and raw_predictions_from_model.shape[1] > 1:
                    logger.info(f"LightGBM booster returned 2D array, shape: {raw_predictions_from_model.shape}. Applying argmax for class labels.")
                    predictions = np.argmax(raw_predictions_from_model, axis=1)
                elif raw_predictions_from_model.ndim == 1:
                    logger.warning(f"LightGBM booster returned 1D array, shape: {raw_predictions_from_model.shape}. Assuming direct class labels or needs thresholding if binary probabilities.")
                    predictions = raw_predictions_from_model 
                else:
                    logger.error(f"Unexpected prediction shape from LightGBM booster: {raw_predictions_from_model.shape}")
                    predictions = raw_predictions_from_model 

            elif isinstance(model, CatBoostClassifier):
                if raw_predictions_from_model.ndim > 1 and raw_predictions_from_model.shape[1] == 1:
                    logger.info(f"CatBoost returned 2D array with 1 col, shape: {raw_predictions_from_model.shape}. Flattening.")
                    predictions = raw_predictions_from_model.flatten()
                elif raw_predictions_from_model.ndim > 1 and raw_predictions_from_model.shape[1] > 1:
                    logger.info(f"CatBoost returned multi-column output for predict, shape: {raw_predictions_from_model.shape}. Applying argmax.")
                    predictions = np.argmax(raw_predictions_from_model, axis=1)
                else: 
                    logger.info(f"CatBoost returned 1D array, shape: {raw_predictions_from_model.shape}.")
                    predictions = raw_predictions_from_model
            
            elif hasattr(model, 'classes_') and raw_predictions_from_model.ndim > 1 and raw_predictions_from_model.shape[1] > 1 : 
                logger.warning(f"Model {model_id_from_request} ({type(model)}) returned multi-column output for .predict(), shape: {raw_predictions_from_model.shape}. Applying argmax.")
                predictions = np.argmax(raw_predictions_from_model, axis=1)
            
            else: 
                logger.info(f"Model {model_id_from_request} ({type(model)}) returned standard 1D output or already processed. Shape: {raw_predictions_from_model.shape if hasattr(raw_predictions_from_model, 'shape') else 'N/A'}")
                predictions = raw_predictions_from_model

            if predictions is not None:
                predictions = np.array(predictions).flatten().astype(int)
                logger.info(f"/predict: Final predictions processed. Shape: {predictions.shape}. Length: {len(predictions)}") # <<< ENHANCED LOGGING
            else:
                logger.error(f"Predictions array is None after model-specific handling for {model_id_from_request}. This indicates an issue.")
                return jsonify({"error": f"Failed to derive class labels from model {model_id_from_request} output."}), 500

            results_df_for_display = input_df.copy() 
            results_df_for_display['Predicted_Class'] = predictions.tolist()
            response_data = results_df_for_display.to_dict(orient='records')
            
            metrics = None
            # --- METRICS CALCULATION DEBUGGING ---
            if TARGET_COLUMN in original_input_df_for_metrics.columns:
                logger.debug(f"/predict METRICS: Target column '{TARGET_COLUMN}' found in original_input_df_for_metrics.") # <<< ENHANCED LOGGING
                # Log some info about the target column before processing
                logger.debug(f"/predict METRICS: original_input_df_for_metrics['{TARGET_COLUMN}'] head:\n{original_input_df_for_metrics[TARGET_COLUMN].head()}") # <<< ENHANCED LOGGING
                logger.debug(f"/predict METRICS: original_input_df_for_metrics['{TARGET_COLUMN}'] dtype: {original_input_df_for_metrics[TARGET_COLUMN].dtype}") # <<< ENHANCED LOGGING
                logger.debug(f"/predict METRICS: original_input_df_for_metrics['{TARGET_COLUMN}'] NaN count: {original_input_df_for_metrics[TARGET_COLUMN].isnull().sum()}") # <<< ENHANCED LOGGING

                try:
                    valid_target_indices = original_input_df_for_metrics[TARGET_COLUMN].dropna().index
                    logger.debug(f"/predict METRICS: valid_target_indices length: {len(valid_target_indices)}") # <<< ENHANCED LOGGING
                    if not valid_target_indices.empty: # <<< ENHANCED LOGGING
                         logger.debug(f"/predict METRICS: valid_target_indices head: {valid_target_indices[:5]}") # <<< ENHANCED LOGGING
                    else: # <<< ENHANCED LOGGING
                        logger.warning("/predict METRICS: valid_target_indices is empty after dropna(). This means all target values were NaN or column was empty.") # <<< ENHANCED LOGGING

                    y_true_series, y_pred_series = pd.Series([], dtype=int), pd.Series([], dtype=int) # Initialize with dtype

                    logger.debug(f"/predict METRICS: Length of 'predictions' array: {len(predictions)}") # <<< ENHANCED LOGGING
                    logger.debug(f"/predict METRICS: Length of 'original_input_df_for_metrics': {len(original_input_df_for_metrics)}") # <<< ENHANCED LOGGING

                    if len(predictions) == len(original_input_df_for_metrics):
                        logger.debug("/predict METRICS: Lengths of predictions and original_input_df match. Proceeding with y_true/y_pred alignment.") # <<< ENHANCED LOGGING
                        
                        # Attempt to convert TARGET_COLUMN to int, log issues
                        try:
                            y_true_all_temp = original_input_df_for_metrics.loc[valid_target_indices, TARGET_COLUMN]
                            logger.debug(f"/predict METRICS: y_true_all_temp (before astype(int)) head:\n{y_true_all_temp.head()}") # <<< ENHANCED LOGGING
                            y_true_all = y_true_all_temp.astype(int)
                            logger.debug(f"/predict METRICS: y_true_all (after astype(int)) head:\n{y_true_all.head()}") # <<< ENHANCED LOGGING
                        except Exception as e_astype:
                            logger.error(f"/predict METRICS: Error converting TARGET_COLUMN to int: {e_astype}", exc_info=True) # <<< ENHANCED LOGGING
                            y_true_all = pd.Series([], dtype=int) # Ensure it's an empty series on error

                        logger.debug(f"/predict METRICS: y_true_all length: {len(y_true_all)}") # <<< ENHANCED LOGGING
                        
                        # Ensure predictions_series has the same index as original_input_df_for_metrics for correct alignment
                        predictions_series_aligned = pd.Series(predictions, index=original_input_df_for_metrics.index)
                        logger.debug(f"/predict METRICS: predictions_series_aligned created. Length: {len(predictions_series_aligned)}") # <<< ENHANCED LOGGING
                        
                        y_pred_all = predictions_series_aligned.loc[valid_target_indices].astype(int)
                        logger.debug(f"/predict METRICS: y_pred_all length: {len(y_pred_all)}") # <<< ENHANCED LOGGING
                        
                        common_indices = y_true_all.index.intersection(y_pred_all.index)
                        logger.debug(f"/predict METRICS: common_indices length: {len(common_indices)}") # <<< ENHANCED LOGGING
                        
                        y_true_series = y_true_all.loc[common_indices]
                        y_pred_series = y_pred_all.loc[common_indices]
                        logger.debug(f"/predict METRICS: Final y_true_series length: {len(y_true_series)}, Final y_pred_series length: {len(y_pred_series)}") # <<< ENHANCED LOGGING
                    else:
                        logger.error("/predict METRICS: Mismatch in length between 'predictions' and 'original_input_df_for_metrics'. Metrics will likely not be calculated with scores.")

                    if not y_true_series.empty and not y_pred_series.empty:
                        logger.info("/predict METRICS: y_true_series and y_pred_series are not empty. Calculating scores.") # <<< ENHANCED LOGGING
                        accuracy = accuracy_score(y_true_series, y_pred_series)
                        precision_macro = precision_score(y_true_series, y_pred_series, average='macro', zero_division=0)
                        recall_macro = recall_score(y_true_series, y_pred_series, average='macro', zero_division=0)
                        f1_macro = f1_score(y_true_series, y_pred_series, average='macro', zero_division=0)
                        
                        cm_labels_unique = sorted(np.unique(np.concatenate((y_true_series.unique(), y_pred_series.unique()))).tolist())
                        if not cm_labels_unique: 
                             cm_labels_unique = [0] 
                        
                        cm = confusion_matrix(y_true_series, y_pred_series, labels=cm_labels_unique)
                        
                        per_class_specificity = []
                        if cm.size > 0 and len(cm_labels_unique) == cm.shape[0] and len(cm_labels_unique) == cm.shape[1]:
                            for i, label_val_cm in enumerate(cm_labels_unique):
                                tp = cm[i, i]
                                fp = cm[:, i].sum() - tp
                                fn = cm[i, :].sum() - tp
                                tn = cm.sum() - (tp + fp + fn)
                                specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                                per_class_specificity.append(specificity_i)
                        else:
                            logger.warning(f"/predict METRICS: CM shape ({cm.shape}) or cm_labels_unique ({len(cm_labels_unique)}) mismatch or CM empty. Specificity might be 0 or inaccurate.")
                            per_class_specificity = [np.nan] * len(cm_labels_unique)

                        specificity_macro = np.nanmean(per_class_specificity) if per_class_specificity else 0.0
                        
                        metrics = {
                            "accuracy": f"{accuracy:.4f}",
                            "precision_macro": f"{precision_macro:.4f}",
                            "sensitivity_macro": f"{recall_macro:.4f}", 
                            "specificity_macro": f"{specificity_macro:.4f}",
                            "f1_macro": f"{f1_macro:.4f}",
                            "confusion_matrix": cm.tolist(),
                            "cm_labels": cm_labels_unique,
                            "message": f"Metrics calculated using '{TARGET_COLUMN}' from uploaded file for model '{model_id_from_request}'."
                        }
                        logger.info(f"/predict METRICS: Metrics calculation complete for {model_id_from_request}. Accuracy: {accuracy:.4f}") # <<< ENHANCED LOGGING
                    else:
                        logger.warning(f"/predict METRICS: y_true_series or y_pred_series is empty after processing. y_true_series length: {len(y_true_series)}, y_pred_series length: {len(y_pred_series)}. Metrics not calculated with detailed scores.") # <<< ENHANCED LOGGING
                        metrics = {"message": f"Target column '{TARGET_COLUMN}' found, but no aligned (y_true, y_pred) data for metric calculation after processing."}
                except Exception as e_metrics:
                    logger.error(f"Error calculating metrics for {model_id_from_request}: {e_metrics}", exc_info=True)
                    metrics = {"error": f"Could not calculate metrics: {str(e_metrics)}"}
            else:
                logger.info(f"/predict METRICS: Target column '{TARGET_COLUMN}' not found in original_input_df_for_metrics. Columns present: {original_input_df_for_metrics.columns.tolist()}. Metrics not calculated.") # <<< ENHANCED LOGGING
                metrics = {"message": f"Target column '{TARGET_COLUMN}' not found in uploaded file. Metrics not calculated."}
            # --- END METRICS CALCULATION DEBUGGING ---

            logger.info(f"/predict: Returning prediction response for {model_id_from_request}.")
            return jsonify({
                "predictions": response_data, 
                "headers": results_df_for_display.columns.tolist(),
                "metrics": metrics,
                "model_used": model_id_from_request
            })

        except pd.errors.EmptyDataError:
            logger.error("/predict: Uploaded CSV file is empty.", exc_info=True)
            return jsonify({"error": "Uploaded CSV file is empty."}), 400
        except Exception as e_process:
            logger.error(f"Error during prediction processing for {model_id_from_request}: {e_process}", exc_info=True)
            return jsonify({"error": f"An error occurred: {str(e_process)}"}), 500
    else:
        logger.warning(f"/predict: Invalid file type received: {file.filename if file else 'No file'}")
        return jsonify({"error": "Invalid file type. Please upload a CSV file."}), 400

if __name__ == '__main__':
    logger.info("--- Flask App Starting (direct run via __main__) ---")
    logger.info(f"--- Local Dev: Models will be loaded from: {MODEL_BASE_DIR} (abs: {os.path.abspath(MODEL_BASE_DIR)}) ---")
    if not os.path.isdir(MODEL_BASE_DIR):
         logger.critical(f"--- Local Dev CRITICAL: MODEL_BASE_DIR does not exist or is not a directory: {os.path.abspath(MODEL_BASE_DIR)} ---")
    else:
        logger.info(f"--- Local Dev SUCCESS: MODEL_BASE_DIR exists: {os.path.abspath(MODEL_BASE_DIR)} ---")
        logger.info(f"--- Local Dev Contents of MODEL_BASE_DIR: {os.listdir(MODEL_BASE_DIR)} ---")

    logger.info(f"--- Local Dev: To run, ensure the 'model_output' directory (from training) is present here or MODEL_BASE_DIR is correctly set.")
    logger.info(f"--- Local Dev: Expected subdirectories in '{MODEL_BASE_DIR}': {', '.join([MODEL_CONFIG[key]['model_subdir'] for key in MODEL_CONFIG])}")
    local_port = int(os.environ.get("PORT", 5001)) 
    logger.info(f"--- Local Dev: Starting Flask development server on host 0.0.0.0, port {local_port} ---")
    app.run(debug=False, host='0.0.0.0', port=local_port)
else:
    logger.info("--- Flask App Initializing (imported by WSGI server like Gunicorn) ---")
    logger.info("--- Gunicorn: Application instance 'app' is ready to be served. ---")

