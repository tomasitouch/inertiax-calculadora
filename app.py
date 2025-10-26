from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import mercadopago
import pandas as pd
import numpy as np
import requests
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
    session,
)
from flask_cors import CORS
from openai import OpenAI
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib import colors
from reportlab.lib.units import inch
import io
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURACIÓN PROFESIONAL
# ==============================

class Config:
    # Seguridad empresarial
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_pro_max_2025_secure_key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "50")) * 1024 * 1024
    SESSION_COOKIE_NAME = "inertiax_pro_session"
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True

    # Archivos profesionales
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_pro")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx", ".xlsm"}

    # Integraciones enterprise
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-calculadora-1.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    MP_PUBLIC_KEY = os.getenv("MP_PUBLIC_KEY")
    
    # AI Configuration - Modelos de última generación
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Sistema de acceso premium
    GUEST_CODES = set(
        (os.getenv("GUEST_CODES") or "INERTIAXVIP2025,ENTRENADORPRO,INVEXORTEST,PREMIUM2025").split(",")
    )

# ==============================
# INICIALIZACIÓN ENTERPRISE
# ==============================

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

# CORS para integraciones empresariales
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Logging profesional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/inertiax_pro.log')
    ]
)
log = logging.getLogger("inertiax_pro")

# Clientes enterprise
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

# ==============================
# MODELOS DE DATOS PROFESIONALES
# ==============================

@dataclass
class AthleteAnalysis:
    name: str
    metrics: Dict
    trends: Dict
    recommendations: List[str]
    performance_score: float

@dataclass
class ExerciseAnalysis:
    name: str
    biomechanics: Dict
    load_progression: Dict
    velocity_profile: Dict
    efficiency_metrics: Dict

@dataclass
class SessionAnalysis:
    date: str
    fatigue_index: float
    performance_quality: float
    technical_consistency: float
    workload: float

def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _job_meta_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "meta.json")

def _save_meta(job_id: str, meta: Dict) -> str:
    p = _job_meta_path(job_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p

def _load_meta(job_id: str) -> Dict:
    p = _job_meta_path(job_id)
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_job() -> str:
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]

def _allowed_file(name: str) -> bool:
    ext = os.path.splitext(name)[1].lower()
    return ext in app.config["ALLOWED_EXT"]

# ==============================
# DETECCIÓN AUTOMÁTICA DE FORMATOS
# ==============================

def detect_file_type(df: pd.DataFrame, filename: str, origin: str) -> Dict:
    """
    Detecta automáticamente el tipo de archivo y formato específico
    """
    detection_result = {
        "origin_app": origin,
        "file_type": "desconocido",
        "sub_type": "desconocido",
        "confidence": 0.0,
        "columns_found": list(df.columns),
        "analysis_required": True
    }
    
    try:
        columns_str = ' '.join(df.columns.astype(str)).lower()
        filename_lower = filename.lower()
        
        # ===========================================
        # DETECCIÓN PARA ENCODER HORIZONTAL
        # ===========================================
        if origin == "encoder_horizontal":
            detection_result["origin_app"] = "encoder_horizontal"
            
            # Patrones específicos del Encoder Horizontal
            horizontal_patterns = {
                'resumen': ['fecha', 'velocidad_max', 'aceleracion_max', 'aceleracion_media', 'tiempo_reaccion', 'velocidad/peso'],
                'velocidad': ['tiempo', 'velocidad'],
                'aceleracion': ['tiempo', 'aceleracion'],
                'distancia': ['tiempo', 'distancia'],
                'velocidad_vs_distancia': ['distancia', 'velocidad']
            }
            
            # Detección por nombre de archivo
            if 'velocidad' in filename_lower and 'distancia' in filename_lower:
                detection_result.update({"file_type": "velocidad_vs_distancia", "confidence": 0.9})
            elif 'velocidad' in filename_lower:
                detection_result.update({"file_type": "velocidad", "confidence": 0.8})
            elif 'aceleracion' in filename_lower or 'aceleración' in filename_lower:
                detection_result.update({"file_type": "aceleracion", "confidence": 0.8})
            elif 'distancia' in filename_lower:
                detection_result.update({"file_type": "distancia", "confidence": 0.8})
            elif 'resumen' in filename_lower or 'sprint_data' in filename_lower:
                detection_result.update({"file_type": "resumen", "confidence": 0.9})
            else:
                # Detección por columnas
                for file_type, pattern_list in horizontal_patterns.items():
                    matches = sum(1 for pattern in pattern_list if pattern in columns_str)
                    if matches >= 2:
                        detection_result.update({"file_type": file_type, "confidence": matches/len(pattern_list)})
                        break
            
            # Detección por estructura
            if detection_result["file_type"] == "desconocido":
                if len(df.columns) >= 6 and any('fecha' in col.lower() for col in df.columns):
                    detection_result.update({"file_type": "resumen", "confidence": 0.7})
                elif len(df.columns) == 2:
                    col1, col2 = df.columns[0].lower(), df.columns[1].lower()
                    if 'tiempo' in col1 and 'velocidad' in col2:
                        detection_result.update({"file_type": "velocidad", "confidence": 0.8})
                    elif 'tiempo' in col1 and 'aceleracion' in col2:
                        detection_result.update({"file_type": "aceleracion", "confidence": 0.8})
                    elif 'tiempo' in col1 and 'distancia' in col2:
                        detection_result.update({"file_type": "distancia", "confidence": 0.8})
                    elif 'distancia' in col1 and 'velocidad' in col2:
                        detection_result.update({"file_type": "velocidad_vs_distancia", "confidence": 0.8})

        # ===========================================
        # DETECCIÓN PARA ENCODER VERTICAL V1
        # ===========================================
        elif origin == "encoder_vertical_v1":
            detection_result["origin_app"] = "encoder_vertical_v1"
            
            # Patrones específicos del Encoder Vertical V1 (PyQt6)
            v1_patterns = ['user', 'exercise', 'type', 'rep_number', 'load', 'max_velocity', 'avg_velocity', 'duration']
            matches = sum(1 for pattern in v1_patterns if pattern in columns_str)
            
            if matches >= 5:
                detection_result.update({
                    "file_type": "encoder_vertical_v1_completo", 
                    "sub_type": "fuerza_velocidad",
                    "confidence": matches/len(v1_patterns)
                })
            elif 'max_velocity' in columns_str and 'avg_velocity' in columns_str:
                detection_result.update({
                    "file_type": "encoder_vertical_v1_resumen", 
                    "sub_type": "metricas_velocidad",
                    "confidence": 0.7
                })
            else:
                detection_result.update({
                    "file_type": "encoder_vertical_v1_generico", 
                    "sub_type": "datos_genericos",
                    "confidence": 0.5
                })

        # ===========================================
        # DETECCIÓN PARA ENCODER VERTICAL ANDROID
        # ===========================================
        elif origin == "app_android_encoder_vertical":
            detection_result["origin_app"] = "app_android_encoder_vertical"
            
            # Patrones del Encoder Vertical Android
            android_patterns = ['athlete', 'exercise', 'date', 'repetition', 'load(kg)', 'concentricvelocity(m/s)', 'eccentricvelocity(m/s)', 'maxvelocity(m/s)']
            matches = sum(1 for pattern in android_patterns if pattern in columns_str)
            
            if matches >= 4:
                detection_result.update({
                    "file_type": "encoder_vertical_android_completo",
                    "sub_type": "biomecanica_completa", 
                    "confidence": matches/len(android_patterns)
                })
            elif any('velocity' in col.lower() for col in df.columns):
                detection_result.update({
                    "file_type": "encoder_vertical_android_velocidad",
                    "sub_type": "analisis_velocidad",
                    "confidence": 0.6
                })
            else:
                detection_result.update({
                    "file_type": "encoder_vertical_android_generico",
                    "sub_type": "datos_entrenamiento", 
                    "confidence": 0.4
                })

        # Validar confianza mínima
        if detection_result["confidence"] < 0.3:
            detection_result["file_type"] = "desconocido"
            detection_result["analysis_required"] = False

        log.info(f"🔍 DETECCIÓN: {detection_result}")

    except Exception as e:
        log.error(f"Error en detección de formato: {str(e)}")
        detection_result["analysis_required"] = False

    return detection_result

# ==============================
# PROCESAMIENTO DE DATOS AVANZADO
# ==============================

def parse_dataframe(path: str) -> pd.DataFrame:
    """Procesamiento profesional de datos con múltiples validaciones"""
    try:
        ext = os.path.splitext(path)[1].lower()
        log.info(f"Procesando archivo: {path} (extensión: {ext})")
        
        if ext == ".csv":
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'windows-1252', 'cp1252']
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, encoding=encoding)
                    log.info(f"Archivo CSV leído con encoding: {encoding}")
                    return df
                except (UnicodeDecodeError, UnicodeError):
                    continue
            return pd.read_csv(path, encoding='utf-8', errors='replace')
        else:
            return pd.read_excel(path)
    except Exception as e:
        log.error(f"Error crítico procesando archivo: {str(e)}")
        raise

def preprocess_data_by_origin(df: pd.DataFrame, origin: str, file_type: str) -> pd.DataFrame:
    """
    Procesamiento científico de datos según el tipo de archivo detectado
    """
    log.info(f"Iniciando procesamiento científico para: {origin} - {file_type}")
    
    # Standardización básica de columnas
    df.columns = [str(col).strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
    
    if origin == "encoder_horizontal":
        return preprocess_encoder_horizontal(df, file_type)
    elif origin == "encoder_vertical_v1":
        return preprocess_encoder_vertical_v1(df, file_type)
    elif origin == "app_android_encoder_vertical":
        return preprocess_encoder_vertical_android(df, file_type)
    else:
        return preprocess_generic_data(df)

def preprocess_encoder_horizontal(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """Procesamiento para Encoder Horizontal"""
    try:
        # Procesamiento según tipo específico
        if file_type == "velocidad":
            if len(df.columns) >= 2:
                df.columns = ["tiempo_s", "velocidad_m_s"]
                df["velocidad_km_h"] = df["velocidad_m_s"] * 3.6
                
        elif file_type == "aceleracion":
            if len(df.columns) >= 2:
                df.columns = ["tiempo_s", "aceleracion_m_s2"]
                
        elif file_type == "distancia":
            if len(df.columns) >= 2:
                df.columns = ["tiempo_s", "distancia_m"]
                # Calcular velocidad instantánea
                if len(df) > 1:
                    df["velocidad_instantanea_m_s"] = df["distancia_m"].diff() / df["tiempo_s"].diff()
                    df["velocidad_instantanea_km_h"] = df["velocidad_instantanea_m_s"] * 3.6
                    
        elif file_type == "velocidad_vs_distancia":
            if len(df.columns) >= 2:
                df.columns = ["distancia_m", "velocidad_m_s"]
                df["velocidad_km_h"] = df["velocidad_m_s"] * 3.6
                
        elif file_type == "resumen":
            # Procesar archivo de resumen con múltiples métricas
            column_mapping = {
                'fecha': 'fecha',
                'velocidad_max(m/s)': 'velocidad_maxima_m_s', 
                'aceleracion_max(m/s2)': 'aceleracion_maxima_m_s2',
                'aceleracion_media(m/s2)': 'aceleracion_media_m_s2',
                'tiempo_reaccion(s)': 'tiempo_reaccion_s',
                'velocidad/peso((m/s)/kg)': 'relacion_velocidad_peso'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns:
                    df[new_col] = pd.to_numeric(df[old_col], errors='coerce')
                    
            if 'velocidad_maxima_m_s' in df.columns:
                df['velocidad_maxima_km_h'] = df['velocidad_maxima_m_s'] * 3.6

        # Conversión numérica para todas las columnas posibles
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='ignore')

        log.info(f"Procesamiento Encoder Horizontal completado: {df.shape}")
        return df

    except Exception as e:
        log.error(f"Error en procesamiento Encoder Horizontal: {str(e)}")
        return df

def preprocess_encoder_vertical_v1(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """Procesamiento para Encoder Vertical V1 (PyQt6)"""
    try:
        # Mapeo de columnas estándar
        column_mapping = {
            'user': 'atleta',
            'exercise': 'ejercicio', 
            'type': 'tipo_ejercicio',
            'rep_number': 'repeticion',
            'load': 'carga_kg',
            'max_velocity': 'velocidad_maxima_m_s',
            'avg_velocity': 'velocidad_media_m_s', 
            'duration': 'duracion_s'
        }
        
        # Aplicar mapeo
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Procesamiento numérico
        numeric_cols = ['carga_kg', 'velocidad_maxima_m_s', 'velocidad_media_m_s', 'duracion_s']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cálculos avanzados para perfil fuerza-velocidad
        if 'carga_kg' in df.columns and 'velocidad_media_m_s' in df.columns:
            df['potencia_w'] = df['carga_kg'] * 9.81 * df['velocidad_media_m_s']  # P = F*v
            df['fuerza_n'] = df['carga_kg'] * 9.81
            
        # Identificar tipo de test
        if 'repeticion' in df.columns:
            if df['repeticion'].astype(str).str.contains('\.').any():
                df['tipo_test'] = 'perfil_fuerza_velocidad'
            else:
                df['tipo_test'] = 'entrenamiento_normal'

        log.info(f"Procesamiento Encoder Vertical V1 completado: {df.shape}")
        return df

    except Exception as e:
        log.error(f"Error en procesamiento Encoder Vertical V1: {str(e)}")
        return df

def preprocess_encoder_vertical_android(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """Procesamiento para Encoder Vertical Android"""
    try:
        # Mapeo profesional de columnas
        rename_map = {
            "Athlete": "atleta",
            "Exercise": "ejercicio",
            "Date": "fecha",
            "Repetition": "repeticion",
            "Load(kg)": "carga_kg",
            "ConcentricVelocity(m/s)": "velocidad_concentrica_m_s",
            "EccentricVelocity(m/s)": "velocidad_eccentrica_m_s", 
            "MaxVelocity(m/s)": "velocidad_maxima_m_s",
            "Duration(s)": "duracion_s",
            "Estimated1RM": "estimado_1rm_kg",
            "Power(W)": "potencia_w",
            "Force(N)": "fuerza_n"
        }
        
        existing_rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=existing_rename_map, inplace=True)
        
        # Procesamiento numérico científico
        numeric_columns = [
            "carga_kg", "velocidad_concentrica_m_s", "velocidad_eccentrica_m_s",
            "velocidad_maxima_m_s", "duracion_s", "estimado_1rm_kg", "potencia_w", "fuerza_n"
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Cálculos biomecánicos avanzados
        if "velocidad_concentrica_m_s" in df.columns and "carga_kg" in df.columns:
            df["potencia_relativa_w_kg"] = df["carga_kg"] * df["velocidad_concentrica_m_s"]
            df["impulso_mecanico"] = df["carga_kg"] * df["velocidad_concentrica_m_s"] * df.get("duracion_s", 1)
        
        if "velocidad_concentrica_m_s" in df.columns and "velocidad_eccentrica_m_s" in df.columns:
            df["ratio_excentrico_concentrico"] = df["velocidad_eccentrica_m_s"] / df["velocidad_concentrica_m_s"]
        
        # Procesamiento temporal profesional
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            df["dia_semana"] = df["fecha"].dt.day_name()
            df["semana_entrenamiento"] = df["fecha"].dt.isocalendar().week
        
        # Métricas de calidad
        if "repeticion" in df.columns:
            df["fatiga_intra_serie"] = df.groupby(["atleta", "ejercicio", "fecha"])["velocidad_concentrica_m_s"].transform(
                lambda x: (x.iloc[0] - x.iloc[-1]) / x.iloc[0] * 100 if len(x) > 1 else 0
            )
        
        # Limpieza científica
        initial_rows = len(df)
        df.dropna(subset=["atleta", "ejercicio"], inplace=True)
        if "carga_kg" in df.columns:
            df = df[df["carga_kg"] > 0]  # Eliminar cargas inválidas
        final_rows = len(df)
        
        log.info(f"Procesamiento Android completado: {initial_rows} -> {final_rows} filas válidas")
        return df

    except Exception as e:
        log.error(f"Error en procesamiento Android: {str(e)}")
        return df

def preprocess_generic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Procesamiento genérico para formatos desconocidos"""
    # Conversión numérica automática
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df

# ==============================
# ANÁLISIS CIENTÍFICO AVANZADO
# ==============================

def generate_comprehensive_stats(df: pd.DataFrame, file_type: str) -> str:
    """Genera estadísticas científicas completas según el tipo de archivo"""
    stats_lines = []
    
    if file_type.startswith("encoder_horizontal"):
        stats_lines.extend(generate_horizontal_stats(df, file_type))
    elif file_type.startswith("encoder_vertical"):
        stats_lines.extend(generate_vertical_stats(df, file_type))
    else:
        stats_lines.extend(generate_generic_stats(df))
    
    return "\n".join(stats_lines)

def generate_horizontal_stats(df: pd.DataFrame, file_type: str) -> List[str]:
    """Estadísticas para Encoder Horizontal"""
    stats = []
    stats.append("🏃 ANÁLISIS ENCODER HORIZONTAL - DATOS DE SPRINT")
    stats.append("=" * 60)
    stats.append(f"• Tipo de datos: {file_type.upper().replace('_', ' ')}")
    stats.append(f"• Total de registros: {df.shape[0]:,}")
    
    if file_type == "velocidad":
        if "velocidad_m_s" in df.columns:
            stats.extend(analyze_velocity_data(df))
    elif file_type == "aceleracion":
        if "aceleracion_m_s2" in df.columns:
            stats.extend(analyze_acceleration_data(df))
    elif file_type == "distancia":
        if "distancia_m" in df.columns:
            stats.extend(analyze_distance_data(df))
    elif file_type == "velocidad_vs_distancia":
        if "velocidad_m_s" in df.columns and "distancia_m" in df.columns:
            stats.extend(analyze_velocity_distance_data(df))
    elif file_type == "resumen":
        stats.extend(analyze_summary_data(df))
    
    return stats

def generate_vertical_stats(df: pd.DataFrame, file_type: str) -> List[str]:
    """Estadísticas para Encoder Vertical"""
    stats = []
    stats.append("🏋️ ANÁLISIS ENCODER VERTICAL - FUERZA Y POTENCIA")
    stats.append("=" * 60)
    stats.append(f"• Tipo de datos: {file_type.upper().replace('_', ' ')}")
    stats.append(f"• Total de registros: {df.shape[0]:,}")
    
    if "atleta" in df.columns:
        stats.append(f"• Atletas únicos: {df['atleta'].nunique()}")
    
    if "ejercicio" in df.columns:
        stats.append(f"• Ejercicios únicos: {df['ejercicio'].nunique()}")
    
    # Métricas específicas de fuerza-velocidad
    if "carga_kg" in df.columns and "velocidad_media_m_s" in df.columns:
        stats.extend(analyze_force_velocity_data(df))
    
    if "velocidad_concentrica_m_s" in df.columns:
        stats.extend(analyze_velocity_metrics(df))
    
    return stats

def generate_generic_stats(df: pd.DataFrame) -> List[str]:
    """Estadísticas genéricas"""
    stats = []
    stats.append("📊 ANÁLISIS ESTADÍSTICO GENERAL")
    stats.append("=" * 60)
    stats.append(f"• Total de registros: {df.shape[0]:,}")
    stats.append(f"• Total de variables: {df.shape[1]}")
    
    # Análisis de columnas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats.append("\n🔢 VARIABLES NUMÉRICAS:")
        for col in numeric_cols[:6]:  # Mostrar máximo 6 columnas
            desc = df[col].describe()
            stats.append(f"• {col}: μ={desc['mean']:.2f} ± {desc['std']:.2f} | Range: {desc['min']:.2f}-{desc['max']:.2f}")
    
    return stats

def analyze_velocity_data(df: pd.DataFrame) -> List[str]:
    """Análisis específico para datos de velocidad"""
    stats = []
    if "velocidad_m_s" in df.columns:
        v = df["velocidad_m_s"]
        stats.append(f"• Velocidad máxima: {v.max():.2f} m/s ({v.max()*3.6:.1f} km/h)")
        stats.append(f"• Velocidad promedio: {v.mean():.2f} m/s ({v.mean()*3.6:.1f} km/h)")
        stats.append(f"• Aceleración promedio: {(v.max() - v.iloc[0]) / len(v) * 10:.2f} m/s²")
    return stats

def analyze_acceleration_data(df: pd.DataFrame) -> List[str]:
    """Análisis específico para datos de aceleración"""
    stats = []
    if "aceleracion_m_s2" in df.columns:
        a = df["aceleracion_m_s2"]
        stats.append(f"• Aceleración máxima: {a.max():.2f} m/s²")
        stats.append(f"• Aceleración promedio: {a.mean():.2f} m/s²")
        stats.append(f"• Fuerza relativa estimada: {a.max() * 75:.0f} N (75kg)")
    return stats

def analyze_force_velocity_data(df: pd.DataFrame) -> List[str]:
    """Análisis de relación fuerza-velocidad"""
    stats = []
    stats.append("\n💪 PERFIL FUERZA-VELOCIDAD:")
    
    carga = df["carga_kg"]
    velocidad = df["velocidad_media_m_s"]
    
    stats.append(f"• Carga máxima: {carga.max():.1f} kg")
    stats.append(f"• Velocidad media: {velocidad.mean():.3f} m/s")
    
    if "potencia_w" in df.columns:
        potencia = df["potencia_w"]
        stats.append(f"• Potencia pico: {potencia.max():.0f} W")
        stats.append(f"• Potencia media: {potencia.mean():.0f} W")
    
    return stats

# ==============================
# IA PROFESIONAL - ANÁLISIS CIENTÍFICO
# ==============================

def run_professional_ai_analysis(df: pd.DataFrame, meta: dict, file_detection: dict) -> dict:
    """
    Análisis científico profesional por IA especializada según el tipo de datos
    """
    if not ai_client:
        return {
            "analysis": "🔬 SERVICIO DE IA NO DISPONIBLE - Configure OPENAI_API_KEY",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": ["Configurar API key de OpenAI para análisis profesional"]
        }
    
    try:
        # Estadísticas específicas según el tipo de archivo
        comprehensive_stats = generate_comprehensive_stats(df, file_detection["file_type"])
        
        # Preparar datos completos para IA
        data_completa = df.to_csv(index=False)
        
        # Contexto científico profesional específico
        contexto = build_analysis_context(meta, file_detection, comprehensive_stats, df)
        
        # Prompt específico según el tipo de datos
        system_prompt = build_system_prompt(file_detection)
        
        user_prompt = f"{contexto}\n```csv\n{data_completa}\n```"

        log.info(f"🧠 INICIANDO ANÁLISIS CIENTÍFICO PARA: {file_detection['file_type']}")
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=8000,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format={"type": "json_object"}
        )
        
        log.info("✅ ANÁLISIS CIENTÍFICO COMPLETADO")
        result = json.loads(response.choices[0].message.content)
        
        # Validación de resultado
        if not all(key in result for key in ["analysis", "python_code_for_charts", "charts_description", "recommendations"]):
            raise ValueError("Respuesta de IA incompleta")
            
        return result
        
    except Exception as e:
        log.error(f"❌ ERROR EN ANÁLISIS CIENTÍFICO: {str(e)}")
        return {
            "analysis": f"❌ ERROR EN ANÁLISIS CIENTÍFICO: {str(e)}",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": ["Contactar soporte técnico para análisis científico"]
        }

def build_analysis_context(meta: dict, file_detection: dict, stats: str, df: pd.DataFrame) -> str:
    """Construye el contexto específico para el análisis"""
    
    origin_descriptions = {
        "encoder_horizontal": "Encoder Horizontal - Análisis de Sprint y Aceleración",
        "encoder_vertical_v1": "Encoder Vertical V1 (PyQt6) - Perfil Fuerza-Velocidad", 
        "app_android_encoder_vertical": "Encoder Vertical Android - Biomecánica Completa"
    }
    
    file_type_descriptions = {
        "velocidad": "Velocidad vs Tiempo en Sprint",
        "aceleracion": "Aceleración vs Tiempo", 
        "distancia": "Distancia vs Tiempo",
        "velocidad_vs_distancia": "Velocidad vs Distancia",
        "resumen": "Resumen de Métricas de Sprint",
        "encoder_vertical_v1_completo": "Datos Completos Fuerza-Velocidad",
        "encoder_vertical_android_completo": "Biomecánica Completa de Levantamiento"
    }
    
    contexto = f"""
ANÁLISIS CIENTÍFICO PROFESIONAL - SISTEMA INERTIAX PRO
===================================================

INFORMACIÓN DEL ANÁLISIS:
• Entrenador: {meta.get('nombre_entrenador', 'Profesional del Deporte')}
• Sistema de origen: {origin_descriptions.get(file_detection['origin_app'], file_detection['origin_app'])}
• Tipo de datos: {file_type_descriptions.get(file_detection['file_type'], file_detection['file_type'])}
• Confianza de detección: {file_detection['confidence']:.1%}
• Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Total de datos procesados: {df.shape[0]:,} registros

ESTADÍSTICAS CIENTÍFICAS COMPLETAS:
{stats}

COLUMNAS DETECTADAS: {file_detection['columns_found']}

INSTRUCCIONES ESPECÍFICAS PARA ANÁLISIS:
"""
    
    # Instrucciones específicas por tipo de datos
    if file_detection["origin_app"] == "encoder_horizontal":
        contexto += """
ANÁLISIS DE SPRINT Y ACELERACIÓN:
1. Evaluar perfil de aceleración y velocidad máxima
2. Analizar técnica de carrera y eficiencia mecánica  
3. Identificar puntos de fatiga y mantenimiento de velocidad
4. Recomendaciones para mejora de aceleración y velocidad
"""
    elif file_detection["origin_app"] == "encoder_vertical_v1":
        contexto += """
ANÁLISIS DE FUERZA-VELOCIDAD:
1. Evaluar perfil individual fuerza-velocidad
2. Analizar relación carga-velocidad con regresiones
3. Identificar zonas óptimas de entrenamiento
4. Recomendaciones para desarrollo de fuerza y potencia
"""
    elif file_detection["origin_app"] == "app_android_encoder_vertical":
        contexto += """
ANÁLISIS BIOMECÁNICO COMPLETO:
1. Evaluar técnica de levantamiento y consistencia
2. Analizar fatiga intra-serie e inter-sesión
3. Identificar asimetrías y desbalances
4. Optimizar prescripción de carga y volumen
"""

    contexto += "\nDATOS COMPLETOS PARA ANÁLISIS CIENTÍFICO:"
    return contexto

def build_system_prompt(file_detection: dict) -> str:
    """Construye el prompt del sistema según el tipo de datos"""
    
    base_prompt = """
Eres un equipo de científicos deportivos con PhD en Biomecánica, Fisiología del Ejercicio y Analítica Deportiva. 
Tienes 20+ años de experiencia en alto rendimiento y investigación científica.

PROTOCOLO DE ANÁLISIS CIENTÍFICO:
1. ANÁLISIS ESPECÍFICO SEGÚN TIPO DE DATOS
2. METODOLOGÍA ESTADÍSTICA RIGUROSA  
3. INTERPRETACIÓN CIENTÍFICA BASADA EN EVIDENCIA
4. COMUNICACIÓN PROFESIONAL Y ACCIONABLE

RESPONDER EN FORMATO JSON ESTRICTAMENTE:
{
    "analysis": "Análisis científico completo...",
    "python_code_for_charts": "Código Python para gráficos profesionales...",
    "charts_description": "Descripción detallada de visualizaciones...", 
    "recommendations": ["Recomendación 1...", "Recomendación 2..."]
}
"""
    
    # Especialización según tipo de datos
    if file_detection["origin_app"] == "encoder_horizontal":
        base_prompt += """
ENFOQUE PARA DATOS DE SPRINT:
- Análisis de curva de aceleración y velocidad
- Identificación de fases del sprint
- Eficiencia mecánica y técnica de carrera
- Métricas de potencia y fatiga
"""
    elif file_detection["origin_app"] == "encoder_vertical":
        base_prompt += """
ENFOQUE PARA DATOS DE FUERZA:
- Perfiles individuales fuerza-velocidad
- Análisis de relación carga-velocidad
- Eficiencia neuromuscular
- Prescripción de entrenamiento óptimo
"""
    
    return base_prompt

# ==============================
# VISUALIZACIONES PROFESIONALES
# ==============================

def generate_professional_charts(df: pd.DataFrame, file_detection: dict) -> List[BytesIO]:
    """
    Genera visualizaciones científicas profesionales según el tipo de datos
    """
    charts = []
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Configuración profesional
        professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        # Gráficos específicos según el tipo de datos
        if file_detection["origin_app"] == "encoder_horizontal":
            charts.extend(generate_horizontal_charts(df, file_detection, professional_colors))
        elif "encoder_vertical" in file_detection["origin_app"]:
            charts.extend(generate_vertical_charts(df, file_detection, professional_colors))
        else:
            charts.extend(generate_generic_charts(df, professional_colors))
            
    except Exception as e:
        log.error(f"Error en generación de gráficos profesionales: {str(e)}")
    
    return charts

def generate_horizontal_charts(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Gráficos específicos para Encoder Horizontal"""
    charts = []
    
    try:
        if file_detection["file_type"] == "velocidad" and "velocidad_m_s" in df.columns:
            # Gráfico de velocidad vs tiempo
            fig, ax = plt.subplots(figsize=(12, 6))
            
            if "tiempo_s" in df.columns:
                ax.plot(df["tiempo_s"], df["velocidad_m_s"], linewidth=2.5, color=colors[0])
                ax.set_xlabel('Tiempo (s)', fontsize=11, fontweight='bold')
            else:
                ax.plot(df.index, df["velocidad_m_s"], linewidth=2.5, color=colors[0])
                ax.set_xlabel('Muestras', fontsize=11, fontweight='bold')
            
            ax.set_ylabel('Velocidad (m/s)', fontsize=11, fontweight='bold')
            ax.set_title('PERFIL DE VELOCIDAD - ANÁLISIS DE SPRINT', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f9fa')
            
            # Añadir métricas clave
            max_vel = df["velocidad_m_s"].max()
            avg_vel = df["velocidad_m_s"].mean()
            ax.axhline(y=max_vel, color='red', linestyle='--', alpha=0.7, label=f'Máx: {max_vel:.2f} m/s')
            ax.axhline(y=avg_vel, color='green', linestyle='--', alpha=0.7, label=f'Prom: {avg_vel:.2f} m/s')
            ax.legend()
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error en gráficos horizontales: {e}")
    
    return charts

def generate_vertical_charts(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Gráficos específicos para Encoder Vertical"""
    charts = []
    
    try:
        # Gráfico de perfil fuerza-velocidad
        if "carga_kg" in df.columns and "velocidad_media_m_s" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if "atleta" in df.columns:
                for i, athlete in enumerate(df['atleta'].unique()[:4]):
                    athlete_data = df[df['atleta'] == athlete]
                    color = colors[i % len(colors)]
                    
                    scatter = ax.scatter(athlete_data['carga_kg'], athlete_data['velocidad_media_m_s'],
                                       c=color, alpha=0.7, s=60, label=athlete, edgecolors='white', linewidth=0.5)
                    
                    # Línea de tendencia
                    if len(athlete_data) > 1:
                        z = np.polyfit(athlete_data['carga_kg'], athlete_data['velocidad_media_m_s'], 1)
                        p = np.poly1d(z)
                        x_range = np.linspace(athlete_data['carga_kg'].min(), athlete_data['carga_kg'].max(), 100)
                        ax.plot(x_range, p(x_range), color=color, linestyle='--', alpha=0.8, linewidth=2)
            else:
                ax.scatter(df['carga_kg'], df['velocidad_media_m_s'], alpha=0.7, s=50, color=colors[0])
            
            ax.set_xlabel('Carga (kg)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Velocidad Media (m/s)', fontsize=11, fontweight='bold')
            ax.set_title('PERFIL FUERZA-VELOCIDAD', fontsize=13, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error en gráficos verticales: {e}")
    
    return charts

def generate_generic_charts(df: pd.DataFrame, colors: list) -> List[BytesIO]:
    """Gráficos genéricos para datos desconocidos"""
    charts = []
    
    try:
        # Gráfico de distribución de variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:3]):
                axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color=colors[i], edgecolor='black')
                axes[i].set_title(f'Distribución de {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error en gráficos genéricos: {e}")
    
    return charts

def save_plot_to_buffer(fig) -> BytesIO:
    """Guarda gráfico en buffer con calidad profesional"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==============================
# GENERACIÓN DE REPORTES PROFESIONALES
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict, file_detection: dict) -> str:
    """Genera reporte PDF profesional"""
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_profesional_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos profesionales personalizados
        title_style = ParagraphStyle(
            'ProfessionalTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1
        )
        
        subtitle_style = ParagraphStyle(
            'ProfessionalSubtitle', 
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#3B1F2B'),
            spaceAfter=12
        )
        
        story = []
        
        # Header profesional
        story.append(Paragraph("REPORTE CIENTÍFICO INERTIAX PRO", title_style))
        story.append(Spacer(1, 10))
        
        # Información del análisis específica
        origin_names = {
            "encoder_horizontal": "Encoder Horizontal - Análisis de Sprint",
            "encoder_vertical_v1": "Encoder Vertical V1 - Fuerza-Velocidad", 
            "app_android_encoder_vertical": "Encoder Vertical Android - Biomecánica"
        }
        
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'Profesional')}<br/>
        <b>Sistema analizado:</b> {origin_names.get(file_detection['origin_app'], file_detection['origin_app'])}<br/>
        <b>Tipo de datos:</b> {file_detection['file_type'].replace('_', ' ').title()}<br/>
        <b>Fecha de generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Confianza de detección:</b> {file_detection['confidence']:.1%}
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Análisis científico
        story.append(Paragraph("ANÁLISIS CIENTÍFICO PROFESIONAL", subtitle_style))
        analysis_text = ai_result.get('analysis', 'Análisis no disponible').replace('\n', '<br/>')
        story.append(Paragraph(analysis_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Gráficos profesionales
        if charts:
            story.append(Paragraph("VISUALIZACIONES CIENTÍFICAS", subtitle_style))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts[:4]):  # Máximo 4 gráficos
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 5))
                    desc = ai_result.get('charts_description', 'Visualización científica').split('.')[i] if i < len(ai_result.get('charts_description', '').split('.')) else 'Gráfico profesional'
                    story.append(Paragraph(f"Figura {i+1}: {desc}", styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Recomendaciones profesionales
        story.append(Paragraph("RECOMENDACIONES CIENTÍFICAS", subtitle_style))
        recommendations = ai_result.get('recommendations', [])
        if isinstance(recommendations, list):
            for rec in recommendations[:8]:  # Máximo 8 recomendaciones
                story.append(Paragraph(f"• {rec}", styles['Normal']))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph(str(recommendations), styles['Normal']))
        
        # Footer profesional
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX Professional Analysis System v3.0<br/>
        Sistema multi-formato para análisis deportivo científico<br/>
        © 2024 InertiaX - Todos los derechos reservados</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF profesional: {str(e)}")
        # Crear PDF de error profesional
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX PRO - ERROR EN REPORTE")
        c.drawString(100, 730, f"Error: {str(e)}")
        c.drawString(100, 710, "Contacte al soporte técnico")
        c.save()
        return error_path

# ==============================
# RUTAS PROFESIONALES ACTUALIZADAS
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "message": "InertiaX Professional API Running",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        # Al refrescar, recuperar el job y mostrar previsualización
        job_id = session.get("job_id")
        if not job_id:
            return render_template("index.html")

        meta = _load_meta(job_id)
        if not meta:
            return render_template("index.html")

        try:
            df = parse_dataframe(meta["file_path"])
            file_detection = detect_file_type(df, meta["file_name"], meta["form"]["origen_app"])
            df = preprocess_data_by_origin(df, meta["form"]["origen_app"], file_detection["file_type"])

            table_html = df.head(15).to_html(
                classes="table table-striped table-bordered table-hover table-sm",
                index=False,
                escape=False
            )

            return render_template(
                "index.html",
                table_html=table_html,
                filename=meta["file_name"],
                form_data=meta["form"],
                file_detection=file_detection,
                mensaje=("🔓 ACCESO PREMIUM ACTIVADO - Análisis profesional disponible"
                        if meta["payment_ok"]
                        else None),
                show_payment=(not meta["payment_ok"]),
            )
        except:
            return render_template("index.html")

    # ===== POST =====
    f = request.files.get("file")
    if not f or f.filename == "":
        return render_template("index.html", error="❌ ARCHIVO NO ESPECIFICADO - Seleccione un archivo para análisis")

    try:
        job_id = _ensure_job()
        session.modified = True

        form = {
            "nombre_entrenador": request.form.get("nombre_entrenador", "").strip(),
            "origen_app": request.form.get("origen_app", "").strip(),
            "codigo_invitado": request.form.get("codigo_invitado", "").strip(),
        }

        log.info(f"📥 Solicitud de análisis profesional de: {form['nombre_entrenador']}")

        code = form.get("codigo_invitado", "")
        payment_ok = False
        mensaje = None
        if code and code in app.config["GUEST_CODES"]:
            payment_ok = True
            mensaje = "🔓 ACCESO PREMIUM ACTIVADO - Análisis profesional disponible"

        ext = os.path.splitext(f.filename)[1].lower()
        safe_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(_job_dir(job_id), safe_name)
        f.save(save_path)

        # Detección automática del tipo de archivo
        df = parse_dataframe(save_path)
        file_detection = detect_file_type(df, f.filename, form["origen_app"])
        
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "payment_ok": payment_ok,
            "form": form,
            "file_detection": file_detection,
            "upload_time": datetime.now().isoformat(),
        }
        _save_meta(job_id, meta)

        # Procesamiento específico
        df = preprocess_data_by_origin(df, form["origen_app"], file_detection["file_type"])

        table_html = df.head(15).to_html(
            classes="table table-striped table-bordered table-hover table-sm",
            index=False,
            escape=False
        )

        return render_template(
            "index.html",
            table_html=table_html,
            filename=f.filename,
            form_data=form,
            file_detection=file_detection,
            mensaje=mensaje,
            show_payment=(not payment_ok),
        )

    except Exception as e:
        log.error(f"❌ Error en procesamiento: {str(e)}")
        return render_template("index.html", error=f"❌ ERROR EN PROCESAMIENTO: {str(e)}")

@app.route("/create_preference", methods=["POST"])
def create_preference():
    """Sistema de pago profesional"""
    if not mp:
        return jsonify(error="SISTEMA DE PAGO NO CONFIGURADO"), 500

    job_id = session.get("job_id")
    if not job_id:
        return jsonify(error="SESIÓN INVÁLIDA"), 400

    # Precio profesional por servicio premium
    price = 990  # Servicio profesional premium

    pref_data = {
        "items": [{
            "title": "InertiaX Pro - Análisis Científico Premium",
            "description": "Análisis biomecánico profesional con IA científica multi-formato",
            "quantity": 1,
            "unit_price": price,
            "currency_id": "CLP",
        }],
        "back_urls": {
            "success": f"{app.config['DOMAIN_URL']}/success",
            "failure": f"{app.config['DOMAIN_URL']}/cancel", 
            "pending": f"{app.config['DOMAIN_URL']}/cancel",
        },
        "auto_return": "approved",
    }

    try:
        pref = mp.preference().create(pref_data)
        return jsonify(pref.get("response", {}))
    except Exception as e:
        log.error(f"Error en sistema de pago: {e}")
        return jsonify(error=str(e)), 500

@app.route("/success")
def success():
    """Pago exitoso - profesional"""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    meta["payment_ok"] = True
    _save_meta(job_id, meta)
    
    log.info(f"✅ Pago exitoso para job: {job_id}")
    return render_template("success.html")

@app.route("/cancel") 
def cancel():
    """Pago cancelado"""
    return render_template("cancel.html")

@app.route("/generate_report")
def generate_report():
    """Generación de reporte profesional completo"""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    if not meta.get("payment_ok"):
        return redirect(url_for("index"))

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="❌ ARCHIVO NO ENCONTRADO")

    try:
        log.info("🚀 INICIANDO GENERACIÓN DE REPORTE PROFESIONAL MULTI-FORMATO")
        
        # 1. Carga y procesamiento profesional
        df = parse_dataframe(file_path)
        file_detection = meta.get("file_detection", {})
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""), file_detection.get("file_type", "desconocido"))
        
        log.info(f"📊 Dataset profesional cargado: {df.shape[0]} registros - Tipo: {file_detection.get('file_type', 'desconocido')}")

        # 2. Análisis científico con IA específico
        log.info("🧠 EJECUTANDO ANÁLISIS CIENTÍFICO ESPECÍFICO...")
        ai_result = run_professional_ai_analysis(df, meta.get("form", {}), file_detection)
        
        # 3. Generación de gráficos profesionales específicos
        log.info("📈 GENERANDO VISUALIZACIONES ESPECÍFICAS...")
        professional_charts = generate_professional_charts(df, file_detection)
        
        # 4. Gráficos adicionales de IA si están disponibles
        ai_charts = []
        python_code = ai_result.get("python_code_for_charts", "")
        if python_code:
            try:
                ai_charts = execute_ai_charts_code(python_code, df)
            except Exception as e:
                log.error(f"Error en gráficos IA: {e}")

        # 5. Generación de reporte PDF profesional específico
        log.info("📄 GENERANDO REPORTE PDF PROFESIONAL...")
        all_charts = professional_charts + ai_charts
        pdf_path = generate_professional_pdf(ai_result, all_charts, meta.get("form", {}), file_detection)

        # 6. Creación de paquete profesional
        zip_path = os.path.join(_job_dir(job_id), f"reporte_profesional_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "INERTIAX_PRO_Reporte_Cientifico.pdf")
            zf.write(file_path, f"datos_originales/{os.path.basename(meta.get('file_name', 'datos.csv'))}")
            
            # Agregar datos procesados
            processed_data_path = os.path.join(_job_dir(job_id), "datos_procesados.csv")
            df.to_csv(processed_data_path, index=False)
            zf.write(processed_data_path, "datos_procesados/analisis_completo.csv")

        # Limpieza profesional
        try:
            os.remove(pdf_path)
            os.remove(processed_data_path)
        except:
            pass

        log.info("✅ REPORTE PROFESIONAL MULTI-FORMATO GENERADO EXITOSAMENTE")
            
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"InertiaX_Pro_Reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ ERROR EN GENERACIÓN DE REPORTE: {str(e)}")
        return render_template("index.html", error=f"❌ ERROR CRÍTICO: {str(e)}")

def execute_ai_charts_code(python_code: str, df: pd.DataFrame) -> List[BytesIO]:
    """Ejecuta código Python de gráficos generado por IA"""
    if not python_code.strip():
        return []
    
    try:
        local_vars = {
            'df': df, 
            'BytesIO': BytesIO, 
            'plt': plt,
            'sns': sns,
            'np': np,
            'pd': pd,
            'charts': []
        }
        
        exec(python_code, local_vars)
        
        charts = local_vars.get('charts', [])
        if not isinstance(charts, list):
            charts = []
            
        return charts
        
    except Exception as e:
        log.error(f"Error ejecutando código de gráficos IA: {e}")
        return []

@app.route("/preview_analysis")
def preview_analysis():
    """Vista previa profesional del análisis"""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    if not meta.get("payment_ok"):
        return redirect(url_for("index"))

    file_path = meta.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return render_template("index.html", error="❌ ARCHIVO NO ENCONTRADO")

    try:
        df = parse_dataframe(file_path)
        file_detection = meta.get("file_detection", {})
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""), file_detection.get("file_type", "desconocido"))
        
        # Análisis científico específico
        ai_result = run_professional_ai_analysis(df, meta.get("form", {}), file_detection)
        
        return render_template(
            "preview.html",
            ai_analysis=ai_result.get("analysis", "Análisis científico en progreso..."),
            recommendations=ai_result.get("recommendations", []),
            filename=meta.get("file_name"),
            file_detection=file_detection
        )
        
    except Exception as e:
        log.error(f"Error en vista previa: {e}")
        return render_template("index.html", error=f"Error en vista previa: {e}")

# ==============================
# MANEJO DE ERRORES PROFESIONAL
# ==============================

@app.errorhandler(413)
def too_large(_e):
    return render_template("index.html", error="❌ ARCHIVO DEMASIADO GRANDE - Máximo 50MB")

@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", error="❌ RECURSO NO ENCONTRADO")

@app.errorhandler(500)
def internal_error(_e):
    return render_template("index.html", error="❌ ERROR INTERNO DEL SERVIDOR")

@app.errorhandler(Exception)
def global_error(e):
    log.exception("Error no controlado en sistema profesional")
    return render_template("index.html", error=f"❌ ERROR DEL SISTEMA: {str(e)}")

# ==============================
# INICIALIZACIÓN PROFESIONAL
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX PROFESSIONAL v3.0 STARTING ON PORT {port}")
    log.info("✅ SISTEMA MULTI-FORMATO ACTIVADO: Encoder Horizontal, Encoder Vertical V1, Encoder Vertical Android")
    app.run(host="0.0.0.0", port=port, debug=False)
