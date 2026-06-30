import sqlite3
import os
import datetime
import hashlib
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import numpy as np
import requests

# --- Constants ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_NAME = os.path.join(BASE_DIR, 'complaints.db')

CV_MODEL_PATH = 'best.pt'

# --- Database Management ---
def init_db():
    """
    Initializes the SQLite database and creates the 'reports' table.
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            timestamp TEXT NOT NULL,
            complaint_text TEXT, 
            category TEXT, 
            gps_lat REAL, 
            gps_lon REAL,
            image_filename TEXT, 
            report_hash TEXT UNIQUE
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"CRITICAL DB ERROR in init_db: {e}")

# --- AI Model Loading ---
@st.cache_resource
def load_models():
    """
    Loads and caches the AI models.
    Uses 'yolov8n.pt' as a fallback if 'best.pt' is not found.
    """
    try:
        if os.path.exists(CV_MODEL_PATH):
            cv_model = YOLO(CV_MODEL_PATH)
            print(f"Production model '{CV_MODEL_PATH}' loaded.")
        else:
            st.error(f"FATAL: '{CV_MODEL_PATH}' not found! Using fallback 'yolov8n.pt'.")
            cv_model = YOLO('yolov8n.pt')

        nlp_model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print("NLP model 'facebook/bart-large-mnli' loaded.")
        return cv_model, nlp_model
    except Exception as e:
        st.error(f"Error loading AI models: {e}")
        return None, None

# --- AI Inference Functions ---
def run_cv_model(cv_model, pil_image):
    """
    Runs the CV model on the image.
    """
    try:
        results = cv_model(pil_image)
        names = cv_model.names

        for box in results[0].boxes:
            detected_class = names[int(box.cls)].lower()
            confidence = float(box.conf)
            
            print(f"CV: Found '{detected_class}' with confidence {confidence:.2f}")

            # Check for our labels with 40% confidence
            if detected_class in ['pothole', 'garbage'] and confidence > 0.4:
                return (True, f"AI Vision: Verified '{detected_class}' detected.")
        
        # Fallback check (for 'yolov8n.pt' test mode)
        # FIX: Re-applied safety check for 'ckpt_path' instead of 'yaml_file'
        model_path = str(getattr(cv_model, 'ckpt_path', ''))
        if 'yolov8n' in model_path and len(results[0].boxes) > 0:
             first_detected = names[int(results[0].boxes[0].cls)].lower()
             return (True, f"AI Vision: TEST MODE (Object Detected: '{first_detected}')")

        return (False, "AI Vision FAILED: No pothole or garbage detected.")
    except Exception as e:
        print(f"CV Error: {e}")
        return (False, f"AI Error: {str(e)}")

def run_nlp_triage(nlp_model, text):
    """
    Runs the NLP model to classify text.
    """
    if not text or text.strip() == "":
        return "Uncategorized"
    try:
        category_labels = ["Road Damage", "Waste Management", "Water Leakage", "Street Light", "Public Nuisance"]
        result = nlp_model(text, candidate_labels=category_labels)
        
        top_category = result['labels'][0] 
        
        if top_category == "Road Damage":
            return "Pothole"
        elif top_category == "Waste Management":
            return "Garbage"
        else:
            return top_category
            
    except Exception as e:
        print(f"NLP Error: {e}")
        return "Uncategorized"

# --- Database Functions ---
def save_report(text, category, lat, lon, img_name):
    """
    Saves a report to the database.
    """
    init_db() 

    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        now = str(datetime.datetime.now())
        data_string = now + text + category + str(lat) + str(lon)
        report_hash = hashlib.sha256(data_string.encode()).hexdigest()
        
        c.execute("INSERT INTO reports (timestamp, complaint_text, category, gps_lat, gps_lon, image_filename, report_hash) VALUES (?,?,?,?,?,?,?)",
                  (now, text, category, lat, lon, img_name, report_hash))
        conn.commit()
        conn.close()
        return (True, "Report logged successfully!")
    except sqlite3.IntegrityError:
        if 'conn' in locals(): conn.close()
        return (False, "Error: This exact report (hash) already exists.")
    except sqlite3.Error as e:
        if 'conn' in locals(): conn.close()
        return (False, str(e)) 

def get_all_reports():
    """
    Gets all reports from the database.
    """
    init_db() 
    try:
        conn = sqlite3.connect(DB_NAME)
        df = pd.read_sql_query("SELECT * FROM reports", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"DB Error in get_all_reports: {e}")
        if 'conn' in locals(): conn.close()
        return pd.DataFrame() 

# --- Predictive AI Functions ---
def get_weather_risk(lat, lon):
    """
    Calls Open-Meteo API for a weather-based risk factor.
    """
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=3&hourly=precipitation"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        precipitation_data = [p for p in data['hourly']['precipitation'] if p is not None]
        total_precipitation = np.sum(precipitation_data)
        
        if total_precipitation > 10.0: return 1.5
        else: return 1.0
    except Exception as e:
        print(f"Weather API Error: {e}")
        return 1.0 

def calculate_risk_score():
    """
    Calculates predictive risk scores based on frequency and weather.
    """
    df = get_all_reports()
    if df.empty:
        return pd.DataFrame(columns=['gps_lat', 'gps_lon', 'risk_score'])

    frequency_df = df.groupby(['gps_lat', 'gps_lon']).size().reset_index(name='frequency')
    weather_factor = get_weather_risk(26.84, 80.94) # Lucknow center
    frequency_df['risk_score'] = frequency_df['frequency'] * weather_factor
    risk_df = frequency_df[frequency_df['risk_score'] > 1.0]
    return risk_df[['gps_lat', 'gps_lon', 'risk_score']] 