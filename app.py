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
# PERSONAL TRAINER PROFESIONAL - ANÁLISIS MEJORADO
# ==============================

def run_professional_ai_analysis(df: pd.DataFrame, meta: dict, file_detection: dict) -> dict:
    """
    Análisis científico profesional como Personal Trainer especializado
    """
    if not ai_client:
        return get_fallback_analysis(df, file_detection)
    
    try:
        # Estadísticas específicas detalladas
        comprehensive_stats = generate_detailed_athlete_stats(df, file_detection)
        
        # Contexto de entrenador personal
        coach_context = build_personal_trainer_context(meta, file_detection, comprehensive_stats, df)
        
        # Prompt especializado como entrenador
        system_prompt = build_personal_trainer_prompt(file_detection)
        
        user_prompt = f"{coach_context}\n```csv\n{df.to_csv(index=False)}\n```"

        log.info(f"🏋️‍♂️ INICIANDO ANÁLISIS COMO PERSONAL TRAINER: {file_detection['file_type']}")
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,  # Más creativo para recomendaciones personalizadas
            max_tokens=12000,  # Más extenso y detallado
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            response_format={"type": "json_object"}
        )
        
        log.info("✅ ANÁLISIS DE PERSONAL TRAINER COMPLETADO")
        result = json.loads(response.choices[0].message.content)
        
        # Enriquecer con análisis adicional
        result = enrich_with_detailed_analysis(result, df, file_detection)
            
        return result
        
    except Exception as e:
        log.error(f"❌ ERROR EN ANÁLISIS PERSONAL TRAINER: {str(e)}")
        return get_fallback_analysis(df, file_detection)

def get_fallback_analysis(df: pd.DataFrame, file_detection: dict) -> dict:
    """Análisis de respaldo cuando la IA no está disponible"""
    
    basic_analysis = f"""
🔬 **ANÁLISIS BÁSICO INERTIAX PRO**
================================

**RESUMEN EJECUTIVO:**
Sistema: {file_detection.get('origin_app', 'Desconocido')}
Tipo de Datos: {file_detection.get('file_type', 'Desconocido')}
Registros Analizados: {df.shape[0]:,}

**DATOS ESTADÍSTICOS:**
{generate_comprehensive_stats(df, file_detection.get('file_type', ''))}

**RECOMENDACIONES GENERALES:**
• Configure su API key de OpenAI para análisis avanzado
• Los datos han sido procesados correctamente
• Considere realizar test periódicos para seguimiento
"""

    return {
        "athlete_profile": "Perfil básico - Active IA para análisis personalizado",
        "technical_analysis": basic_analysis,
        "strengths_weaknesses": {
            "strengths": ["Datos procesados correctamente", "Formato compatible detectado"],
            "weaknesses": ["Análisis limitado sin IA", "Recomendaciones genéricas"],
            "opportunities": ["Activar análisis IA para personalización completa"]
        },
        "training_recommendations": [
            "Configure OPENAI_API_KEY para análisis avanzado",
            "Mantenga consistencia en la toma de datos",
            "Realice evaluaciones periódicas para seguimiento"
        ],
        "python_code_for_charts": "",
        "charts_description": "Gráficos profesionales generados automáticamente"
    }

def build_personal_trainer_context(meta: dict, file_detection: dict, stats: str, df: pd.DataFrame) -> str:
    """Construye contexto detallado de entrenador personal"""
    
    athlete_profile = analyze_athlete_profile(df, file_detection)
    
    contexto = f"""
🎯 **ANÁLISIS PERSONAL TRAINER INERTIAX PRO - EVALUACIÓN COMPLETA**
================================================================

**INFORMACIÓN DEL ATLETA Y ENTRENADOR:**
• Entrenador Personal: {meta.get('nombre_entrenador', 'Especialista Inertiax Pro')}
• Sistema Analizado: {get_system_description(file_detection['origin_app'])}
• Tipo de Entrenamiento: {get_training_type_description(file_detection['file_type'])}
• Fecha de Evaluación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**PERFIL DEL DEPORTISTA:**
{athlete_profile}

**ESTADÍSTICAS DEPORTIVAS DETALLADAS:**
{stats}

**DATOS TÉCNICOS:**
• Total de Sesiones/Registros: {df.shape[0]:,}
• Confianza de Datos: {file_detection['confidence']:.1%}
• Columnas Analizadas: {file_detection['columns_found']}

**INSTRUCCIONES ESPECÍFICAS COMO PERSONAL TRAINER:**
"""

    # Instrucciones específicas por deporte
    if file_detection["origin_app"] == "encoder_horizontal":
        contexto += """
ANÁLISIS DE VELOCISTA - SPRINT Y ACELERACIÓN:
1. Evaluar técnica de salida y aceleración inicial
2. Analizar mantenimiento de velocidad máxima
3. Identificar patrones de fatiga técnica
4. Prescribir ejercicios correctivos específicos
5. Planificar progresión de cargas y volúmenes
"""
    elif "encoder_vertical" in file_detection["origin_app"]:
        contexto += """
ANÁLISIS DE FUERZA-POTENCIA - LEVANTAMIENTOS:
1. Evaluar perfil neuromuscular individual
2. Analizar técnica de ejecución por repetición
3. Identificar desbalances y asimetrías
4. Prescribir ejercicios complementarios
5. Optimizar periodización del entrenamiento
"""

    contexto += "\n**DATOS COMPLETOS PARA ANÁLISIS PERSONALIZADO:**"
    return contexto

def build_personal_trainer_prompt(file_detection: dict) -> str:
    """Prompt especializado como Personal Trainer"""
    
    base_prompt = """
Eres un **Personal Trainer Elite certificado** con especialización en biomecánica, nutrición deportiva y periodización del entrenamiento. Tienes 15+ años trabajando con atletas de alto rendimiento.

**PROTOCOLO DE EVALUACIÓN PERSONALIZADA:**
1. **ANÁLISIS TÉCNICO INDIVIDUAL** - Evaluación movimiento por movimiento
2. **IDENTIFICACIÓN DE FORTALEZAS Y DEBILIDADES** - Análisis comparativo con estándares
3. **RECOMENDACIONES ESPECÍFICAS** - Ejercicios, técnicas y correcciones
4. **PLANIFICACIÓN PERSONALIZADA** - Periodización y progresiones
5. **NUTRICIÓN Y RECUPERACIÓN** - Estrategias complementarias

**FORMATO DE RESPUESTA ESTRICTO EN JSON:**
{
    "athlete_profile": "Perfil completo del deportista...",
    "technical_analysis": "Análisis técnico detallado...",
    "strengths_weaknesses": {
        "strengths": ["Fuerza 1", "Fuerza 2"],
        "weaknesses": ["Debilidad 1", "Debilidad 2"],
        "opportunities": ["Oportunidad 1", "Oportunidad 2"]
    },
    "performance_metrics": {
        "current_level": "Avanzado",
        "progress_trend": "Positivo", 
        "comparison_standards": "Comparación con estándares..."
    },
    "training_recommendations": [
        {
            "category": "Fuerza",
            "exercises": ["Ejercicio 1", "Ejercicio 2"],
            "prescription": "3x8-12, 2min descanso",
            "rationale": "Fundamento científico..."
        }
    ],
    "nutrition_advice": ["Consejo 1", "Consejo 2"],
    "recovery_strategies": ["Estrategia 1", "Estrategia 2"],
    "progress_monitoring": "Indicadores de progreso...",
    "python_code_for_charts": "Código para gráficos...",
    "charts_description": "Descripción de visualizaciones..."
}
"""

    # Especialización adicional según deporte
    if file_detection["origin_app"] == "encoder_horizontal":
        base_prompt += """
**ENFOQUE PARA VELOCISTAS:**
- Análisis de fases del sprint: salida, aceleración, velocidad máxima, deceleración
- Evaluación de técnica de carrera y economía de movimiento
- Prescripción de ejercicios de pliometría y técnica
- Estrategias para mejora de aceleración y velocidad máxima
"""
    elif "encoder_vertical" in file_detection["origin_app"]:
        base_prompt += """
**ENFOQUE PARA DEPORTES DE FUERZA:**
- Análisis de curva fuerza-velocidad-potencia
- Evaluación de técnica de levantamiento
- Detección de sticking points
- Estrategias para superar mesetas
"""

    return base_prompt

def analyze_athlete_profile(df: pd.DataFrame, file_detection: dict) -> str:
    """Genera perfil detallado del atleta"""
    
    profile = []
    
    try:
        if file_detection["origin_app"] == "encoder_horizontal":
            if "velocidad_m_s" in df.columns:
                vel_max = df["velocidad_m_s"].max()
                vel_avg = df["velocidad_m_s"].mean()
                
                # Clasificación de nivel
                if vel_max > 10.0:
                    nivel = "🏆 ELITE - Nivel Mundial"
                elif vel_max > 9.0:
                    nivel = "⭐ AVANZADO - Competitivo"
                elif vel_max > 8.0:
                    nivel = "📈 INTERMEDIO - En Desarrollo"
                else:
                    nivel = "🎯 PRINCIPIANTE - Base Técnica"
                
                profile.append(f"• Nivel de Velocista: {nivel}")
                profile.append(f"• Velocidad Máxima: {vel_max:.2f} m/s ({vel_max*3.6:.1f} km/h)")
                profile.append(f"• Velocidad Promedio: {vel_avg:.2f} m/s ({vel_avg*3.6:.1f} km/h)")
                
                # Análisis de aceleración
                if len(df) > 10:
                    primeros_10 = df["velocidad_m_s"].iloc[:10].mean()
                    profile.append(f"• Aceleración Inicial: {primeros_10:.2f} m/s (primeros 10 puntos)")
        
        elif "encoder_vertical" in file_detection["origin_app"]:
            if "carga_kg" in df.columns and "velocidad_media_m_s" in df.columns:
                carga_max = df["carga_kg"].max()
                vel_promedio = df["velocidad_media_m_s"].mean()
                
                # Clasificación de fuerza
                if carga_max > 150:
                    nivel_fuerza = "🏋️‍♂️ FUERZA ELITE"
                elif carga_max > 100:
                    nivel_fuerza = "💪 FUERZA AVANZADA"
                elif carga_max > 70:
                    nivel_fuerza = "📊 FUERZA INTERMEDIA"
                else:
                    nivel_fuerza = "🎯 FUERZA BASE"
                
                profile.append(f"• Nivel de Fuerza: {nivel_fuerza}")
                profile.append(f"• Carga Máxima Registrada: {carga_max:.1f} kg")
                profile.append(f"• Velocidad Media Ejecución: {vel_promedio:.3f} m/s")
                
                if "potencia_w" in df.columns:
                    potencia_max = df["potencia_w"].max()
                    profile.append(f"• Potencia Pico: {potencia_max:.0f} W")
        
        # Análisis de consistencia
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            consistencia = df[numeric_cols[0]].std() / df[numeric_cols[0]].mean() if df[numeric_cols[0]].mean() != 0 else 0
            profile.append(f"• Coeficiente de Variación: {consistencia:.2%}")
            
    except Exception as e:
        log.error(f"Error en análisis de perfil: {e}")
        profile.append("• Perfil: Análisis en progreso...")
    
    return "\n".join(profile) if profile else "• Perfil: Datos insuficientes para análisis completo"

def enrich_with_detailed_analysis(result: dict, df: pd.DataFrame, file_detection: dict) -> dict:
    """Enriquece el análisis con datos adicionales"""
    
    # Añadir métricas básicas si no están presentes
    if 'performance_metrics' not in result:
        result['performance_metrics'] = {
            'data_quality': 'Alta' if file_detection.get('confidence', 0) > 0.7 else 'Media',
            'records_analyzed': df.shape[0],
            'analysis_date': datetime.now().isoformat()
        }
    
    return result

def generate_detailed_athlete_stats(df: pd.DataFrame, file_detection: dict) -> str:
    """Genera estadísticas detalladas del atleta"""
    return generate_comprehensive_stats(df, file_detection.get('file_type', ''))

def get_system_description(system: str) -> str:
    """Descripciones profesionales de los sistemas"""
    descriptions = {
        "encoder_horizontal": "Encoder Horizontal - Análisis de Sprint y Aceleración",
        "encoder_vertical_v1": "Encoder Vertical V1 - Perfil Fuerza-Velocidad", 
        "app_android_encoder_vertical": "Encoder Vertical Android - Biomecánica Completa"
    }
    return descriptions.get(system, system)

def get_training_type_description(file_type: str) -> str:
    """Descripciones profesionales de tipos de entrenamiento"""
    descriptions = {
        "velocidad": "Análisis de Velocidad y Técnica de Carrera",
        "aceleracion": "Evaluación de Aceleración y Potencia",
        "distancia": "Análisis de Distancia y Economía de Movimiento",
        "resumen": "Evaluación Integral de Métricas",
        "encoder_vertical_v1_completo": "Perfil Completo Fuerza-Velocidad-Potencia"
    }
    return descriptions.get(file_type, file_type.replace('_', ' ').title())

# ==============================
# VISUALIZACIONES PROFESIONALES MEJORADAS
# ==============================

def generate_professional_charts(df: pd.DataFrame, file_detection: dict) -> List[BytesIO]:
    """
    Genera visualizaciones científicas profesionales y estéticas
    """
    charts = []
    
    # Configuración de estilo profesional
    plt.style.use('seaborn-v0_8-whitegrid')
    professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#1A535C']
    
    try:
        # Gráfico 1: Análisis de tendencia principal
        charts.extend(generate_main_trend_analysis(df, file_detection, professional_colors))
        
        # Gráfico 2: Análisis de distribución
        charts.extend(generate_distribution_analysis(df, file_detection, professional_colors))
        
        # Gráfico 3: Análisis comparativo
        charts.extend(generate_comparative_analysis(df, file_detection, professional_colors))
        
        # Gráfico 4: Análisis de progresión temporal
        charts.extend(generate_temporal_analysis(df, file_detection, professional_colors))
        
        # Gráfico 5: Análisis de correlaciones
        charts.extend(generate_correlation_analysis(df, file_detection, professional_colors))
        
    except Exception as e:
        log.error(f"Error en generación de gráficos profesionales: {str(e)}")
    
    return charts

def generate_main_trend_analysis(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Gráfico principal de tendencia con estilo profesional"""
    charts = []
    
    try:
        if file_detection["origin_app"] == "encoder_horizontal":
            # Gráfico de velocidad con múltiples métricas
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('ANÁLISIS COMPLETO DE SPRINT - INERTIAX PRO', fontsize=16, fontweight='bold', y=0.95)
            
            # Subgráfico 1: Velocidad vs Tiempo
            if "velocidad_m_s" in df.columns:
                tiempo = df["tiempo_s"] if "tiempo_s" in df.columns else df.index
                ax1.plot(tiempo, df["velocidad_m_s"], linewidth=2.5, color=colors[0], label='Velocidad (m/s)')
                ax1.set_title('PERFIL DE VELOCIDAD', fontweight='bold')
                ax1.set_xlabel('Tiempo (s)')
                ax1.set_ylabel('Velocidad (m/s)')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Añadir zonas de interés
                max_vel_idx = df["velocidad_m_s"].idxmax()
                ax1.axvline(x=tiempo[max_vel_idx] if hasattr(tiempo, 'iloc') else max_vel_idx, 
                           color='red', linestyle='--', alpha=0.7, label='Velocidad Máxima')
            
            # Subgráfico 2: Aceleración
            if "aceleracion_m_s2" in df.columns:
                ax2.plot(tiempo, df["aceleracion_m_s2"], linewidth=2.5, color=colors[1], label='Aceleración (m/s²)')
                ax2.set_title('PERFIL DE ACELERACIÓN', fontweight='bold')
                ax2.set_xlabel('Tiempo (s)')
                ax2.set_ylabel('Aceleración (m/s²)')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            
            # Subgráfico 3: Potencia estimada
            if "velocidad_m_s" in df.columns:
                masa_estimada = 75  # kg
                potencia = masa_estimada * df["velocidad_m_s"] * df.get("aceleracion_m_s2", 1)
                ax3.plot(tiempo, potencia, linewidth=2.5, color=colors[2], label='Potencia (W)')
                ax3.set_title('POTENCIA ESTIMADA', fontweight='bold')
                ax3.set_xlabel('Tiempo (s)')
                ax3.set_ylabel('Potencia (W)')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            
            # Subgráfico 4: Fase del sprint
            if "distancia_m" in df.columns and "velocidad_m_s" in df.columns:
                ax4.scatter(df["distancia_m"], df["velocidad_m_s"], alpha=0.7, s=50, color=colors[3])
                ax4.set_title('VELOCIDAD vs DISTANCIA', fontweight='bold')
                ax4.set_xlabel('Distancia (m)')
                ax4.set_ylabel('Velocidad (m/s)')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error en gráfico principal: {e}")
    
    return charts

def generate_correlation_analysis(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Análisis de correlaciones con heatmap profesional"""
    charts = []
    
    try:
        # Seleccionar solo columnas numéricas
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calcular matriz de correlación
            corr_matrix = numeric_df.corr()
            
            # Heatmap profesional
            im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Configurar ejes
            ax.set_xticks(np.arange(len(corr_matrix.columns)))
            ax.set_yticks(np.arange(len(corr_matrix.columns)))
            ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr_matrix.columns)
            
            # Añadir valores de correlación
            for i in range(len(corr_matrix.columns)):
                for j in range(len(corr_matrix.columns)):
                    text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            ax.set_title('MATRIZ DE CORRELACIONES - ANÁLISIS MULTIVARIADO', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Barra de color profesional
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Coeficiente de Correlación', rotation=270, labelpad=20)
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error en análisis de correlaciones: {e}")
    
    return charts

def generate_distribution_analysis(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Análisis de distribución de variables clave"""
    charts = []
    
    try:
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
        log.error(f"Error en análisis de distribución: {e}")
    
    return charts

def generate_comparative_analysis(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Análisis comparativo entre atletas o sesiones"""
    charts = []
    
    try:
        if "atleta" in df.columns and df['atleta'].nunique() > 1:
            # Comparativa entre atletas
            fig, ax = plt.subplots(figsize=(12, 6))
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                metric = numeric_cols[0]  # Primera métrica numérica
                
                athlete_means = df.groupby('atleta')[metric].mean().sort_values(ascending=False)
                athlete_means.plot(kind='bar', ax=ax, color=colors[:len(athlete_means)])
                
                ax.set_title(f'COMPARATIVA ENTRE ATLETAS - {metric.upper()}', fontweight='bold')
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                charts.append(save_plot_to_buffer(fig))
                
    except Exception as e:
        log.error(f"Error en análisis comparativo: {e}")
    
    return charts

def generate_temporal_analysis(df: pd.DataFrame, file_detection: dict, colors: list) -> List[BytesIO]:
    """Análisis de progresión temporal"""
    charts = []
    
    try:
        if "fecha" in df.columns:
            df_temp = df.copy()
            df_temp['fecha'] = pd.to_datetime(df_temp['fecha'])
            df_temp = df_temp.sort_values('fecha')
            
            numeric_cols = df_temp.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                metric = numeric_cols[0]  # Primera métrica numérica
                ax.plot(df_temp['fecha'], df_temp[metric], marker='o', linewidth=2, color=colors[0])
                
                ax.set_title(f'PROGRESIÓN TEMPORAL - {metric.upper()}', fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_xlabel('Fecha')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                charts.append(save_plot_to_buffer(fig))
                
    except Exception as e:
        log.error(f"Error en análisis temporal: {e}")
    
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
# GENERACIÓN DE REPORTES PROFESIONALES MEJORADOS
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict, file_detection: dict) -> str:
    """Genera reporte PDF ultra-profesional con análisis personalizado"""
    
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_pt_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos profesionales mejorados
        title_style = ParagraphStyle(
            'ProfessionalTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#3B1F2B'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header profesional mejorado
        story.append(Paragraph("INERTIAX PRO - REPORTE PERSONAL TRAINER", title_style))
        story.append(Spacer(1, 10))
        
        # Información detallada del análisis
        info_text = f"""
        <b>Entrenador Personal:</b> {meta.get('nombre_entrenador', 'Especialista Inertiax Pro')}<br/>
        <b>Sistema Analizado:</b> {get_system_description(file_detection['origin_app'])}<br/>
        <b>Tipo de Evaluación:</b> {get_training_type_description(file_detection['file_type'])}<br/>
        <b>Fecha de Evaluación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Confianza del Análisis:</b> {file_detection['confidence']:.1%}<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Perfil del Atleta
        if 'athlete_profile' in ai_result:
            story.append(Paragraph("👤 PERFIL DEL DEPORTISTA", section_style))
            story.append(Paragraph(ai_result['athlete_profile'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Análisis Técnico
        if 'technical_analysis' in ai_result:
            story.append(Paragraph("🔬 ANÁLISIS TÉCNICO DETALLADO", section_style))
            analysis_text = ai_result['technical_analysis'].replace('\n', '<br/>')
            story.append(Paragraph(analysis_text, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Fortalezas y Debilidades
        if 'strengths_weaknesses' in ai_result:
            story.append(Paragraph("💪 FORTALEZAS Y ÁREAS DE MEJORA", section_style))
            sw = ai_result['strengths_weaknesses']
            
            if 'strengths' in sw and sw['strengths']:
                story.append(Paragraph("<b>Fortalezas Principales:</b>", styles['Normal']))
                for strength in sw['strengths'][:5]:
                    story.append(Paragraph(f"✓ {strength}", styles['Normal']))
                    story.append(Spacer(1, 3))
            
            if 'weaknesses' in sw and sw['weaknesses']:
                story.append(Paragraph("<b>Áreas de Mejora:</b>", styles['Normal']))
                for weakness in sw['weaknesses'][:5]:
                    story.append(Paragraph(f"⚠ {weakness}", styles['Normal']))
                    story.append(Spacer(1, 3))
        
        # Gráficos profesionales
        if charts:
            story.append(Paragraph("📊 VISUALIZACIONES CIENTÍFICAS", section_style))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts[:6]):  # Máximo 6 gráficos
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 5))
                    
                    # Descripción personalizada
                    desc = f"Figura {i+1}: Análisis profesional de {get_chart_description(i, file_detection)}"
                    story.append(Paragraph(desc, styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Recomendaciones de Entrenamiento
        if 'training_recommendations' in ai_result:
            story.append(Paragraph("🎯 PLAN DE ENTRENAMIENTO PERSONALIZADO", section_style))
            recommendations = ai_result['training_recommendations']
            
            if isinstance(recommendations, list):
                for rec in recommendations[:8]:
                    if isinstance(rec, dict):
                        category = rec.get('category', 'General')
                        exercises = ', '.join(rec.get('exercises', []))[:4]
                        story.append(Paragraph(f"<b>{category}:</b> {exercises}", styles['Normal']))
                    else:
                        story.append(Paragraph(f"• {rec}", styles['Normal']))
                    story.append(Spacer(1, 5))
        
        # Footer profesional
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX Professional Personal Trainer System v4.0<br/>
        Análisis biomecánico personalizado con inteligencia artificial<br/>
        © 2024 InertiaX - Tecnología para el Alto Rendimiento</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF profesional: {str(e)}")
        return create_error_pdf(str(e))

def create_error_pdf(error_msg: str) -> str:
    """Crea un PDF de error profesional"""
    error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(error_path)
    c.drawString(100, 750, "INERTIAX PRO - ERROR EN REPORTE")
    c.drawString(100, 730, f"Error: {error_msg}")
    c.drawString(100, 710, "Contacte al soporte técnico")
    c.save()
    return error_path

def get_chart_description(index: int, file_detection: dict) -> str:
    """Descripción personalizada para cada gráfico"""
    descriptions = {
        0: "tendencias principales y evolución",
        1: "distribución de métricas clave", 
        2: "análisis comparativo de rendimiento",
        3: "progresión temporal del desempeño",
        4: "correlaciones entre variables",
        5: "análisis específico del deporte"
    }
    return descriptions.get(index, "análisis profesional")

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
    """Vista previa profesional del análisis - Personal Trainer"""
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
        
        # Análisis científico específico como Personal Trainer
        ai_result = run_professional_ai_analysis(df, meta.get("form", {}), file_detection)
        
        return render_template(
            "preview.html",
            ai_analysis=ai_result.get("technical_analysis", "Análisis científico en progreso..."),
            athlete_profile=ai_result.get("athlete_profile", "Perfil en análisis..."),
            strengths_weaknesses=ai_result.get("strengths_weaknesses", {}),
            training_recommendations=ai_result.get("training_recommendations", []),
            performance_metrics=ai_result.get("performance_metrics", {}),
            nutrition_advice=ai_result.get("nutrition_advice", []),
            recovery_strategies=ai_result.get("recovery_strategies", []),
            filename=meta.get("file_name"),
            file_detection=file_detection,
            form_data=meta.get("form", {}),
            upload_time=meta.get("upload_time", datetime.now().isoformat())
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
    log.info(f"🚀 INERTIAX PROFESSIONAL v4.0 STARTING ON PORT {port}")
    log.info("✅ SISTEMA PERSONAL TRAINER ACTIVADO: Análisis profesional multi-formato")
    app.run(host="0.0.0.0", port=port, debug=False)
