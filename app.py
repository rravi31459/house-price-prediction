import os
import json
import logging
from pathlib import Path
import sys
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.predict import Predictor
from src.train import train_model
from src.utils import generate_prediction_pdf

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "house_price_predictor_secret_key_129837")

# Global variables
HISTORY_FILE = config.DATA_DIR / "prediction_history.json"

def load_prediction_history() -> list:
    """Loads prediction history from a local JSON file."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read prediction history file: {str(e)}")
        return []

def save_prediction_history(history: list):
    """Saves prediction history to a local JSON file, keeping only the last 50 items."""
    try:
        # Limit history size to 50
        history_clipped = history[-50:]
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_clipped, f, indent=4)
    except Exception as e:
        logger.error(f"Failed to write prediction history file: {str(e)}")

def get_latest_metadata() -> dict:
    """Loads the latest model training metadata, or runs training if metadata is missing."""
    metadata_path = config.MODEL_DIR / "metadata_latest.json"
    if not metadata_path.exists():
        logger.warning("Latest metadata not found. Auto-triggering initial model training...")
        try:
            return train_model()
        except Exception as e:
            logger.critical(f"Initial model training failed: {str(e)}")
            return {}
            
    try:
        with open(metadata_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to read latest metadata file: {str(e)}")
        return {}

# Ensure a model is ready when the server starts
_ = get_latest_metadata()

@app.route("/")
def home():
    """Renders the landing page with prediction form and metadata overview."""
    metadata = get_latest_metadata()
    metrics = metadata.get("metrics", {})
    return render_template(
        "index.html", 
        metrics=metrics, 
        currency=config.CURRENCY_SYMBOL,
        features=config.NUMERICAL_FEATURES
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for running single predictions asynchronously."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided."}), 400
            
        sqft = data.get("sqft")
        bedrooms = data.get("bedrooms")
        bathrooms = data.get("bathrooms")
        
        # Validations
        if sqft is None or bedrooms is None or bathrooms is None:
            return jsonify({"success": False, "error": "All fields (Square Footage, Bedrooms, Bathrooms) are required."}), 400
            
        try:
            sqft = float(sqft)
            bedrooms = int(bedrooms)
            bathrooms = float(bathrooms)
        except ValueError:
            return jsonify({"success": False, "error": "Inputs must be numeric values."}), 400
            
        if sqft <= 0 or bedrooms < 0 or bathrooms < 0:
            return jsonify({"success": False, "error": "Values cannot be negative, and square footage must be greater than zero."}), 400
            
        # Run Prediction
        predictor = Predictor()
        result = predictor.predict_single(sqft, bedrooms, bathrooms)
        
        # Save input features into result for PDF generation later
        result["sqft"] = sqft
        result["bedrooms"] = bedrooms
        result["bathrooms"] = bathrooms
        result["id"] = result["timestamp"].replace("-", "").replace(":", "").replace(" ", "-")
        
        # Append to prediction history
        history = load_prediction_history()
        history.append(result)
        save_prediction_history(history)
        
        return jsonify({
            "success": True, 
            "price": result["estimated_price"],
            "currency": config.CURRENCY_SYMBOL,
            "confidence_message": result["confidence_message"],
            "input_summary": result["input_summary"],
            "timestamp": result["timestamp"],
            "id": result["id"]
        })
        
    except FileNotFoundError as fnf:
        return jsonify({"success": False, "error": str(fnf)}), 500
    except Exception as e:
        logger.error(f"Prediction route encountered error: {str(e)}")
        return jsonify({"success": False, "error": "An error occurred on the server while running the prediction."}), 500

@app.route("/model-info")
def model_info():
    """Renders the detailed model details and performance metrics."""
    metadata = get_latest_metadata()
    metrics = metadata.get("metrics", {})
    params = metadata.get("parameters", {})
    coefs = params.get("coefficients", {})
    intercept = params.get("intercept", 0.0)
    train_size = metadata.get("train_samples", 0)
    val_size = metadata.get("validation_samples", 0)
    timestamp = metadata.get("timestamp", "Unknown")
    
    # Render metadata page
    return render_template(
        "model_info.html",
        metrics=metrics,
        coefficients=coefs,
        intercept=intercept,
        train_size=train_size,
        val_size=val_size,
        timestamp=timestamp,
        currency=config.CURRENCY_SYMBOL
    )

@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Handles CSV uploads for batch predictions."""
    if request.method == "POST":
        # Handle file upload
        if "file" not in request.files:
            flash("No file part in the request.", "error")
            return redirect(request.url)
            
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)
            
        if not file.filename.endswith(".csv"):
            flash("Invalid file format. Please upload a valid CSV file.", "error")
            return redirect(request.url)
            
        try:
            # Save uploaded file temporarily in scratch or data dir
            upload_dir = config.DATA_DIR / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            input_path = upload_dir / "temp_input.csv"
            file.save(str(input_path))
            
            # Predict
            predictor = Predictor()
            output_path = upload_dir / "predictions_output.csv"
            predictions_df = predictor.predict_batch(input_path, output_path)
            
            # Convert preview to HTML table
            # Exclude standard ID, but take first 10 rows
            preview_cols = ["Id"] + list(config.COLUMN_MAPPING.keys())[:-1] + ["predicted_price"]
            # Fallback if names are different
            available_cols = [col for col in preview_cols if col in predictions_df.columns]
            if not available_cols:
                available_cols = predictions_df.columns[:5]
                
            preview_html = predictions_df[available_cols].head(10).to_html(
                classes="preview-table", index=False, border=0
            )
            
            # Save total count
            total_predictions = len(predictions_df)
            
            return render_template(
                "batch.html", 
                preview_table=preview_html, 
                total_count=total_predictions, 
                has_results=True
            )
            
        except KeyError as ke:
            flash(f"Mapping Error: {str(ke)}. Ensure the CSV column headers map correctly.", "error")
            logger.error(f"Batch mapping columns missing: {str(ke)}")
        except Exception as e:
            flash(f"An error occurred during batch processing: {str(e)}", "error")
            logger.error(f"Batch prediction error: {str(e)}")
            
        return redirect(request.url)
        
    return render_template("batch.html", has_results=False)

@app.route("/download-batch")
def download_batch():
    """Downloads the batch prediction output CSV file."""
    output_path = config.DATA_DIR / "uploads" / "predictions_output.csv"
    if not output_path.exists():
        flash("No prediction results available to download.", "error")
        return redirect(url_for("batch"))
        
    return send_file(
        str(output_path),
        mimetype="text/csv",
        as_attachment=True,
        download_name="house_price_predictions.csv"
    )

@app.route("/history")
def history():
    """Displays the list of past predictions."""
    history = load_prediction_history()
    # Reverse to show newest predictions first
    history_sorted = list(reversed(history))
    return render_template("history.html", history=history_sorted, currency=config.CURRENCY_SYMBOL)

@app.route("/history/clear", methods=["POST"])
def clear_history():
    """Clears the local prediction history."""
    save_prediction_history([])
    flash("Prediction history has been successfully cleared.", "success")
    return redirect(url_for("history"))

@app.route("/download-pdf/<prediction_id>")
def download_pdf(prediction_id):
    """Generates and downloads a prediction certificate PDF."""
    history = load_prediction_history()
    target_prediction = None
    for record in history:
        if record.get("id") == prediction_id:
            target_prediction = record
            break
            
    if not target_prediction:
        flash("Valuation record not found.", "error")
        return redirect(url_for("history"))
        
    # Set up PDF path
    pdf_dir = config.DATA_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"valuation_{prediction_id}.pdf"
    
    try:
        generate_prediction_pdf(target_prediction, pdf_path)
        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"valuation_report_{prediction_id}.pdf"
        )
    except Exception as e:
        logger.error(f"PDF download failed: {str(e)}")
        flash("Failed to generate PDF valuation report.", "error")
        return redirect(url_for("history"))

@app.route("/retrain", methods=["POST"])
def retrain():
    """Retrains the model on train.csv and refreshes the pipeline."""
    logger.info("Retraining requested by user...")
    try:
        metadata = train_model(config.TRAIN_PATH)
        flash(f"Model successfully retrained! Validation R²: {metadata.get('metrics', {}).get('R2', 0.0):.4f}", "success")
    except Exception as e:
        logger.error(f"Retraining failed: {str(e)}")
        flash(f"Failed to retrain model: {str(e)}", "error")
        
    return redirect(url_for("model_info"))

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 page handler."""
    return render_template("404.html"), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
