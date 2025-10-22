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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ==============================
# CONFIGURACIÓN EMPRESARIAL PROFESIONAL
# ==============================

class Config:
    # Seguridad empresarial
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_enterprise_pro_2025_secure_key")
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH_MB", "100")) * 1024 * 1024
    SESSION_COOKIE_NAME = "inertiax_enterprise_session"
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True

    # Archivos empresariales
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_enterprise")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx", ".xlsm", ".json"}

    # Integraciones enterprise
    DOMAIN_URL = os.getenv("DOMAIN_URL", "https://inertiax-enterprise.onrender.com")
    MP_ACCESS_TOKEN = os.getenv("MP_ACCESS_TOKEN")
    MP_PUBLIC_KEY = os.getenv("MP_PUBLIC_KEY")
    
    # AI Configuration - Modelos de última generación
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Sistema de acceso premium enterprise
    GUEST_CODES = set(
        (os.getenv("GUEST_CODES") or "INERTIAXENTERPRISE2025,COACH_PRO,V1WINDOWSPRO,ANDROIDPRO,PREMIUM2025").split(",")
    )

    # Configuraciones específicas por dispositivo
    DEVICE_PROFILES = {
        "encoder_v1_windows": {
            "name": "Encoder V1 Windows",
            "columns": {
                "user": "user", "exercise": "exercise", "type": "type",
                "rep_number": "rep_number", "load": "load", 
                "max_velocity": "max_velocity", "avg_velocity": "avg_velocity",
                "duration": "duration"
            },
            "analysis_focus": ["fuerza_velocidad", "fatiga_intra_serie", "consistencia_tecnica"]
        },
        "encoder_vertical_android": {
            "name": "Encoder Vertical Android", 
            "columns": {
                "Athlete": "atleta", "Exercise": "ejercicio", "Date": "fecha",
                "Repetition": "repeticion", "Load(kg)": "carga_kg",
                "ConcentricVelocity(m/s)": "velocidad_concentrica_m_s",
                "EccentricVelocity(m/s)": "velocidad_eccentrica_m_s",
                "MaxVelocity(m/s)": "velocidad_maxima_m_s",
                "Duration(s)": "duracion_s", "Estimated1RM": "estimado_1rm_kg",
                "Power(W)": "potencia_w", "Force(N)": "fuerza_n"
            },
            "analysis_focus": ["biomecanica_completa", "potencia", "trabajo_mecanico"]
        },
        "generic_csv": {
            "name": "CSV Genérico",
            "columns": {},
            "analysis_focus": ["analisis_general", "patrones", "tendencias"]
        }
    }

# ==============================
# INICIALIZACIÓN ENTERPRISE
# ==============================

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]

# CORS para integraciones empresariales
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Logging profesional empresarial
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/inertiax_enterprise.log')
    ]
)
log = logging.getLogger("inertiax_enterprise")

# Clientes enterprise
mp = mercadopago.SDK(app.config["MP_ACCESS_TOKEN"]) if app.config["MP_ACCESS_TOKEN"] else None
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

# ==============================
# MODELOS DE DATOS ENTERPRISE
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

@dataclass
class DeviceProfile:
    name: str
    columns: Dict
    analysis_focus: List[str]

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
# PROCESAMIENTO UNIVERSAL DE DATOS
# ==============================

def detect_device_profile(df: pd.DataFrame, origin: str) -> str:
    """Detecta automáticamente el perfil del dispositivo basado en las columnas"""
    origin = origin.lower()
    
    # Detección por origen específico
    if "windows" in origin and "encoder" in origin:
        return "encoder_v1_windows"
    elif "android" in origin and "vertical" in origin:
        return "encoder_vertical_android"
    
    # Detección automática por columnas
    columns_set = set(df.columns.str.lower())
    
    # Verificar columnas de Encoder V1 Windows
    v1_columns = {"user", "exercise", "type", "rep_number", "load", "max_velocity", "avg_velocity", "duration"}
    if v1_columns.issubset(columns_set):
        return "encoder_v1_windows"
    
    # Verificar columnas de Encoder Vertical Android
    android_columns = {"athlete", "exercise", "date", "repetition", "load(kg)", "concentricvelocity(m/s)"}
    if any(col in ' '.join(columns_set) for col in android_columns):
        return "encoder_vertical_android"
    
    return "generic_csv"

def parse_dataframe(path: str) -> pd.DataFrame:
    """Procesamiento universal de datos con múltiples validaciones"""
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
        elif ext == ".json":
            return pd.read_json(path)
        else:
            return pd.read_excel(path)
    except Exception as e:
        log.error(f"Error crítico procesando archivo: {str(e)}")
        raise

def preprocess_data_universal(df: pd.DataFrame, device_profile: str) -> pd.DataFrame:
    """Procesamiento universal de datos para cualquier dispositivo"""
    log.info(f"Iniciando procesamiento universal para: {device_profile}")
    
    profile = app.config["DEVICE_PROFILES"][device_profile]
    rename_map = profile["columns"]
    
    # Estandarizar nombres de columnas
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    # Aplicar mapeo específico del dispositivo
    existing_rename_map = {}
    for old_col, new_col in rename_map.items():
        old_col_lower = old_col.lower().replace(" ", "_")
        if old_col_lower in df.columns:
            existing_rename_map[old_col_lower] = new_col
    
    if existing_rename_map:
        df.rename(columns=existing_rename_map, inplace=True)
        log.info(f"Columnas renombradas: {existing_rename_map}")
    
    # Procesamiento numérico universal
    numeric_columns = [
        "load", "carga_kg", "max_velocity", "avg_velocity", "velocidad_maxima_m_s",
        "velocidad_concentrica_m_s", "velocidad_promedio_m_s", "duration", "duracion_s",
        "rep_number", "repeticion", "estimado_1rm_kg", "potencia_w", "fuerza_n"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Cálculos avanzados universales
    if "load" in df.columns and "avg_velocity" in df.columns:
        df["potencia_instantanea_w"] = df["load"] * 9.81 * df["avg_velocity"]
    
    if "carga_kg" in df.columns and "velocidad_concentrica_m_s" in df.columns:
        df["potencia_relativa_w_kg"] = df["carga_kg"] * df["velocidad_concentrica_m_s"]
        df["impulso_mecanico_ns"] = df["carga_kg"] * 9.81 * df["velocidad_concentrica_m_s"] * df.get("duracion_s", 1)
    
    # Procesamiento temporal
    date_columns = [col for col in df.columns if 'fecha' in col or 'date' in col or 'time' in col]
    if date_columns:
        for date_col in date_columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df["fecha_analisis"] = df[date_col].dt.strftime('%Y-%m-%d')
                df["hora_analisis"] = df[date_col].dt.strftime('%H:%M:%S')
                break
            except:
                continue
    
    # Identificar tipo de análisis
    if "load" in df.columns and "avg_velocity" in df.columns:
        unique_loads = df["load"].nunique()
        if unique_loads > 3:
            df["tipo_analisis"] = "perfil_fuerza_velocidad"
        else:
            df["tipo_analisis"] = "repeticiones_individuales"
    
    # Limpieza final
    initial_rows = len(df)
    if "load" in df.columns:
        df = df[df["load"] > 0]  # Eliminar cargas inválidas
    elif "carga_kg" in df.columns:
        df = df[df["carga_kg"] > 0]
    
    if "avg_velocity" in df.columns:
        df = df[df["avg_velocity"] > 0.1]  # Velocidades mínimas realistas
    elif "velocidad_concentrica_m_s" in df.columns:
        df = df[df["velocidad_concentrica_m_s"] > 0.1]
        
    final_rows = len(df)
    
    log.info(f"Procesamiento completado: {initial_rows} -> {final_rows} filas válidas")
    return df

# ==============================
# ANÁLISIS CIENTÍFICO UNIVERSAL - OPTIMIZADO
# ==============================

def perform_comprehensive_analysis(df: pd.DataFrame, device_profile: str) -> Dict:
    """
    Realiza TODOS los análisis: exploratorio, gráfico, predictivo, reporte e interpretativo
    OPTIMIZADO: Análisis más rápido y eficiente
    """
    analysis_results = {
        "device_profile": device_profile,
        "exploratory_analysis": "",
        "graphical_analysis": "",
        "predictive_model": "",
        "session_report": "",
        "interpretive_analysis": "",
        "advanced_biomechanical": "",
        "charts": []
    }
    
    try:
        # 1. ANÁLISIS EXPLORATORIO (Más rápido)
        analysis_results["exploratory_analysis"] = perform_exploratory_analysis(df, device_profile)
        
        # 2. ANÁLISIS GRÁFICO (Limitado para evitar timeout)
        graphical_results = perform_graphical_analysis(df, device_profile)
        analysis_results["graphical_analysis"] = graphical_results["analysis"]
        analysis_results["charts"].extend(graphical_results["charts"])
        
        # 3. MODELO PREDICTIVO (Simplificado)
        analysis_results["predictive_model"] = perform_predictive_analysis(df)
        
        # 4. REPORTE DE SESIÓN (Rápido)
        analysis_results["session_report"] = generate_session_report(df, device_profile)
        
        # 5. ANÁLISIS INTERPRETATIVO (IA - Opcional, no bloqueante)
        try:
            analysis_results["interpretive_analysis"] = perform_interpretive_analysis(df, device_profile)
        except Exception as e:
            analysis_results["interpretive_analysis"] = f"⚠️ Análisis IA no disponible: {str(e)}"
        
        # 6. ANÁLISIS BIOMECÁNICO AVANZADO (Simplificado)
        analysis_results["advanced_biomechanical"] = perform_advanced_biomechanical_analysis(df)
        
    except Exception as e:
        log.error(f"Error en análisis completo: {str(e)}")
        analysis_results["error"] = f"Error en análisis: {str(e)}"
    
    return analysis_results

def perform_exploratory_analysis(df: pd.DataFrame, device_profile: str) -> str:
    """🧠 1. Análisis exploratorio completo y universal - OPTIMIZADO"""
    analysis_lines = []
    
    profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
    analysis_lines.append(f"🧠 ANÁLISIS EXPLORATORIO - {profile_name}")
    analysis_lines.append("=" * 70)
    
    # Estadísticas básicas universales - MÁS RÁPIDO
    analysis_lines.append("\n📊 ESTADÍSTICAS DESCRIPTIVAS UNIVERSALES:")
    analysis_lines.append("-" * 50)
    
    # Columnas numéricas comunes a todos los dispositivos
    numeric_columns = []
    for col in ['load', 'carga_kg', 'max_velocity', 'avg_velocity', 'velocidad_maxima_m_s', 
                'velocidad_concentrica_m_s', 'duration', 'duracion_s']:
        if col in df.columns:
            numeric_columns.append(col)
    
    for col in numeric_columns[:6]:  # Limitar a 6 columnas máximo
        try:
            stats_desc = df[col].describe()
            cv = (df[col].std() / df[col].mean() * 100) if df[col].mean() > 0 else 0
            analysis_lines.append(f"{col}: μ={stats_desc['mean']:.3f} ± {stats_desc['std']:.3f} "
                               f"(CV={cv:.1f}%) | Range: {stats_desc['min']:.1f}-{stats_desc['max']:.1f}")
        except:
            analysis_lines.append(f"{col}: Error en cálculo")
    
    # Patrones carga vs velocidad (compatible con ambos sistemas) - SIMPLIFICADO
    analysis_lines.append("\n🔍 PATRONES CARGA-VELOCIDAD UNIVERSALES:")
    analysis_lines.append("-" * 45)
    
    load_col = 'load' if 'load' in df.columns else 'carga_kg'
    velocity_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
    
    if load_col in df.columns and velocity_col in df.columns:
        try:
            correlation = df[load_col].corr(df[velocity_col])
            analysis_lines.append(f"Correlación carga-velocidad: {correlation:.3f}")
            
            # Muestreo para análisis más rápido
            sample_df = df.sample(min(100, len(df)), random_state=42)
            load_bins = pd.cut(sample_df[load_col], bins=3)  # Menos bins para más velocidad
            velocity_by_load = sample_df.groupby(load_bins)[velocity_col].agg(['mean', 'std', 'count'])
            for bin_range, stats in velocity_by_load.iterrows():
                analysis_lines.append(f"Carga {bin_range}: Vel {stats['mean']:.3f} ± {stats['std']:.3f} m/s (n={stats['count']})")
        except Exception as e:
            analysis_lines.append(f"Error en análisis carga-velocidad: {str(e)}")
    
    # Análisis por usuario/atleta - LIMITADO
    user_col = 'user' if 'user' in df.columns else 'atleta'
    if user_col in df.columns:
        analysis_lines.append(f"\n👥 ANÁLISIS POR {user_col.upper()} (TOP 5):")
        analysis_lines.append("-" * 40)
        
        user_counts = df[user_col].value_counts()
        top_users = user_counts.head(5).index
        
        for user in top_users:
            user_data = df[df[user_col] == user]
            analysis_lines.append(f"• {user}: {len(user_data)} reps")
    
    return "\n".join(analysis_lines)

def perform_graphical_analysis(df: pd.DataFrame, device_profile: str) -> Dict:
    """📈 2. Análisis gráfico universal automático - OPTIMIZADO"""
    charts = []
    analysis_text = []
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        analysis_text.append(f"📈 ANÁLISIS GRÁFICO UNIVERSAL - {profile_name}")
        analysis_text.append("=" * 60)
        
        # Configuración de columnas universales
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        avg_vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        user_col = 'user' if 'user' in df.columns else 'atleta'
        
        # 1. SOLO UN GRÁFICO PRINCIPAL para evitar timeout
        if load_col in df.columns and max_vel_col in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))  # Tamaño más pequeño
            
            # Muestreo para gráficos más rápidos
            sample_df = df.sample(min(200, len(df)), random_state=42)
            
            # Colores profesionales
            professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
            
            if user_col in sample_df.columns:
                users = sample_df[user_col].unique()[:3]  # Máximo 3 usuarios
                for i, user in enumerate(users):
                    user_data = sample_df[sample_df[user_col] == user]
                    ax.scatter(user_data[load_col], user_data[max_vel_col], 
                              c=professional_colors[i % len(professional_colors)], 
                              alpha=0.7, s=40, label=user)
            else:
                ax.scatter(sample_df[load_col], sample_df[max_vel_col], 
                          alpha=0.6, s=30, color=professional_colors[0])
            
            ax.set_xlabel('Carga (kg)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Velocidad Máxima (m/s)', fontsize=10, fontweight='bold')
            ax.set_title(f'PERFIL FUERZA-VELOCIDAD\n{profile_name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(('fuerza_velocidad_universal', buf))
            analysis_text.append("• Gráfico 1: Perfil Fuerza-Velocidad - Relación fundamental carga-velocidad")
        
        analysis_text.append("\n📋 GRÁFICOS GENERADOS EXITOSAMENTE")
        
    except Exception as e:
        analysis_text.append(f"❌ Error en análisis gráfico universal: {str(e)}")
    
    return {
        "analysis": "\n".join(analysis_text),
        "charts": charts
    }

def perform_predictive_analysis(df: pd.DataFrame) -> str:
    """⚙️ 3. Modelo predictivo universal de velocidad máxima - SIMPLIFICADO"""
    analysis_lines = []
    
    analysis_lines.append("⚙️ MODELO PREDICTIVO UNIVERSAL - VELOCIDAD MÁXIMA")
    analysis_lines.append("=" * 60)
    
    try:
        # Preparar datos universalmente
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        
        if load_col not in df.columns or max_vel_col not in df.columns:
            return "❌ Datos insuficientes para modelo predictivo"
        
        # Muestreo para modelo más rápido
        sample_df = df.sample(min(500, len(df)), random_state=42)
        
        X = sample_df[[load_col]]
        y = sample_df[max_vel_col]
        
        # Modelo simple y rápido
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        analysis_lines.append(f"\n📐 MODELO LINEAL SIMPLE:")
        analysis_lines.append(f"- R²: {r2:.4f}")
        analysis_lines.append(f"- RMSE: {rmse:.4f} m/s")
        analysis_lines.append(f"- Pendiente: {model.coef_[0]:.4f} m/s/kg")
        analysis_lines.append(f"- Intercepto: {model.intercept_:.3f} m/s")
        
        # Interpretación simple
        analysis_lines.append(f"\n💡 INTERPRETACIÓN:")
        analysis_lines.append(f"- El modelo explica el {r2*100:.1f}% de la variación en velocidad")
        analysis_lines.append(f"- Error típico: ±{rmse:.3f} m/s")
        analysis_lines.append(f"- Por cada kg de carga, la velocidad cambia {model.coef_[0]:.4f} m/s")
        
    except Exception as e:
        analysis_lines.append(f"❌ Error en modelo predictivo: {str(e)}")
    
    return "\n".join(analysis_lines)

def generate_session_report(df: pd.DataFrame, device_profile: str) -> str:
    """📊 4. Reporte automatizado de sesión universal - RÁPIDO"""
    report_lines = []
    
    profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
    report_lines.append(f"📊 REPORTE AUTOMATIZADO DE SESIÓN - {profile_name}")
    report_lines.append("=" * 70)
    
    try:
        # Configuración de columnas universales
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        user_col = 'user' if 'user' in df.columns else 'atleta'
        
        # Resumen general de la sesión
        report_lines.append("\n📈 RESUMEN GENERAL DE LA SESIÓN:")
        report_lines.append("-" * 40)
        report_lines.append(f"• Total de repeticiones: {len(df):,}")
        report_lines.append(f"• Usuarios/Atletas: {df[user_col].nunique() if user_col in df.columns else 1}")
        
        # Métricas pico de rendimiento
        if max_vel_col in df.columns:
            peak_velocity = df[max_vel_col].max()
            report_lines.append(f"• Velocidad máxima: {peak_velocity:.3f} m/s")
        
        if load_col in df.columns:
            peak_load = df[load_col].max()
            total_volume = df[load_col].sum()
            report_lines.append(f"• Carga máxima: {peak_load:.1f} kg")
            report_lines.append(f"• Volumen total: {total_volume:.0f} kg")
        
        # Recomendaciones rápidas
        report_lines.append("\n💡 RECOMENDACIONES RÁPIDAS:")
        report_lines.append("-" * 35)
        
        if load_col in df.columns and max_vel_col in df.columns:
            avg_velocity = df[max_vel_col].mean()
            if avg_velocity < 0.5:
                report_lines.append("• ENFOQUE: Trabajar velocidad con cargas ligeras")
            elif avg_velocity > 0.8:
                report_lines.append("• ENFOQUE: Mantener cargas para desarrollo de potencia")
            else:
                report_lines.append("• ENFOQUE: Zona óptima de entrenamiento")
        
    except Exception as e:
        report_lines.append(f"❌ Error en reporte de sesión: {str(e)}")
    
    return "\n".join(report_lines)

def perform_interpretive_analysis(df: pd.DataFrame, device_profile: str) -> str:
    """💬 5. Análisis interpretativo universal (IA explicativa) - NO BLOQUEANTE"""
    
    if not ai_client:
        return "🔍 ANÁLISIS INTERPRETATIVO NO DISPONIBLE - Configure OPENAI_API_KEY"
    
    try:
        # Preparar resumen MUY breve para IA (más rápido)
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        
        data_summary = f"""
        RESUMEN EJECUTIVO PARA ANÁLISIS RÁPIDO:
        - Dispositivo: {app.config["DEVICE_PROFILES"][device_profile]["name"]}
        - Repeticiones: {len(df)}
        - Carga promedio: {df[load_col].mean() if load_col in df.columns else 'N/A':.1f} kg
        - Velocidad promedio: {df[max_vel_col].mean() if max_vel_col in df.columns else 'N/A':.3f} m/s
        - Muestra: {df.head(3).to_string()}
        """
        
        prompt = f"""
        Como entrenador experto, analiza brevemente estos datos de entrenamiento:
        {data_summary}
        
        Proporciona 3-4 conclusiones prácticas en lenguaje simple.
        """
        
        # LLAMADA MÁS RÁPIDA con menos tokens
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "Eres un entrenador conciso y práctico."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,  # Menos tokens para respuesta más rápida
            timeout=30  # Timeout para evitar bloqueos
        )
        
        return f"💬 ANÁLISIS INTERPRETATIVO RÁPIDO\n\n{response.choices[0].message.content}"
        
    except Exception as e:
        return f"⚠️ Análisis IA no disponible: {str(e)}"

def perform_advanced_biomechanical_analysis(df: pd.DataFrame) -> str:
    """🔬 6. Análisis biomecánico avanzado universal - SIMPLIFICADO"""
    analysis_lines = []
    
    analysis_lines.append("🔬 ANÁLISIS BIOMECÁNICO AVANZADO UNIVERSAL")
    analysis_lines.append("=" * 60)
    
    try:
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        
        if load_col not in df.columns or vel_col not in df.columns:
            return "❌ Datos insuficientes para análisis biomecánico"
        
        # Análisis simple de relación fuerza-velocidad
        correlation = df[load_col].corr(df[vel_col])
        analysis_lines.append(f"\n📐 RELACIÓN FUERZA-VELOCIDAD:")
        analysis_lines.append(f"- Correlación: {correlation:.3f}")
        analysis_lines.append(f"- Tipo: {'Inversa' if correlation < -0.3 else 'Directa' if correlation > 0.3 else 'Débil'}")
        
        # Zonas de entrenamiento simples
        avg_velocity = df[vel_col].mean()
        analysis_lines.append(f"\n🎯 ZONA DE ENTRENAMIENTO DETECTADA:")
        if avg_velocity < 0.5:
            analysis_lines.append("- PREDOMINIO: Fuerza máxima (>85% 1RM)")
        elif avg_velocity > 0.8:
            analysis_lines.append("- PREDOMINIO: Velocidad (<70% 1RM)") 
        else:
            analysis_lines.append("- PREDOMINIO: Potencia (70-85% 1RM)")
        
    except Exception as e:
        analysis_lines.append(f"❌ Error en análisis biomecánico: {str(e)}")
    
    return "\n".join(analysis_lines)

# ==============================
# FUNCIONES DE APOYO UNIVERSALES - OPTIMIZADAS
# ==============================

def save_plot_to_buffer(fig) -> BytesIO:
    """Guarda gráfico en buffer con calidad profesional - OPTIMIZADO"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',  # DPI reducido
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_comprehensive_pdf(analysis_results: Dict, meta: dict) -> str:
    """Genera PDF UNIVERSAL con TODOS los análisis - OPTIMIZADO"""
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_universal_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos empresariales personalizados
        title_style = ParagraphStyle(
            'EnterpriseTitle',
            parent=styles['Heading1'],
            fontSize=16,  # Tamaño reducido
            textColor=colors.HexColor('#1a5276'),
            spaceAfter=15,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header empresarial
        story.append(Paragraph("INERTIAX ENTERPRISE - REPORTE UNIVERSAL", title_style))
        story.append(Spacer(1, 10))
        
        # Información del análisis
        device_profile = analysis_results.get("device_profile", "generic_csv")
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'Profesional')}<br/>
        <b>Dispositivo:</b> {profile_name}<br/>
        <b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Sistema:</b> Análisis Universal Optimizado
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Solo las secciones principales
        sections = [
            ("🧠 ANÁLISIS EXPLORATORIO", "exploratory_analysis"),
            ("📊 REPORTE DE SESIÓN", "session_report"),
            ("🔬 ANÁLISIS BIOMECÁNICO", "advanced_biomechanical")
        ]
        
        for title, key in sections:
            content = analysis_results.get(key, "No disponible").replace('\n', '<br/>')
            story.append(Paragraph(f"<b>{title}</b>", styles['Heading2']))
            story.append(Paragraph(content, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX Enterprise Analysis System<br/>
        © 2024 InertiaX Enterprise - Todos los derechos reservados</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF optimizado: {str(e)}")
        # PDF de error mínimo
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX ENTERPRISE - REPORTE")
        c.drawString(100, 730, "Error en generación, contacte soporte")
        c.save()
        return error_path

def generate_word_report(analysis_results: Dict, meta: dict) -> str:
    """Genera reporte en formato Word universal - OPTIMIZADO"""
    try:
        word_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_universal_{uuid.uuid4().hex}.html")
        
        device_profile = analysis_results.get("device_profile", "generic_csv")
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        
        with open(word_path, 'w', encoding='utf-8') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>InertiaX Enterprise - Reporte Universal</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #1a5276; text-align: center; }
                    h2 { color: #2e86ab; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                    .section { margin-bottom: 20px; }
                </style>
            </head>
            <body>
            """)
            
            f.write(f"<h1>INERTIAX ENTERPRISE - REPORTE UNIVERSAL</h1>")
            f.write(f"<p><strong>Entrenador:</strong> {meta.get('nombre_entrenador', 'Profesional')}</p>")
            f.write(f"<p><strong>Dispositivo:</strong> {profile_name}</p>")
            f.write(f"<p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")
            
            # Solo contenido esencial
            essential_sections = [
                ("ANÁLISIS EXPLORATORIO", "exploratory_analysis"),
                ("REPORTE DE SESIÓN", "session_report"), 
                ("ANÁLISIS BIOMECÁNICO", "advanced_biomechanical")
            ]
            
            for title, key in essential_sections:
                content = analysis_results.get(key, "No disponible").replace('\n', '<br/>')
                f.write(f"<div class='section'>")
                f.write(f"<h2>{title}</h2>")
                f.write(f"<div>{content}</div>")
                f.write("</div>")
            
            f.write("</body></html>")
        
        return word_path
        
    except Exception as e:
        log.error(f"Error generando Word optimizado: {str(e)}")
        return ""

# ==============================
# RUTAS ENTERPRISE MEJORADAS - CORREGIDAS
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "message": "InertiaX Enterprise Universal API Running",
        "version": "3.0.0",
        "supported_devices": list(app.config["DEVICE_PROFILES"].keys()),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/upload", methods=["POST"])
def upload():
    """Endpoint enterprise universal para carga de datos - CORREGIDO"""
    try:
        job_id = _ensure_job()
        session.modified = True

        # Datos del formulario enterprise
        form = {
            "nombre_entrenador": request.form.get("nombre_entrenador", "").strip(),
            "origen_app": request.form.get("origen_app", "").strip(),
            "codigo_invitado": request.form.get("codigo_invitado", "").strip(),
        }

        log.info(f"📥 Solicitud de análisis enterprise de: {form['nombre_entrenador']}")

        # Verificación de código premium enterprise
        code = form.get("codigo_invitado", "")
        payment_ok = False
        mensaje = None
        if code and code in app.config["GUEST_CODES"]:
            payment_ok = True
            mensaje = "🔓 ACCESO ENTERPRISE ACTIVADO - Análisis universal disponible"

        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", error="❌ ARCHIVO NO ESPECIFICADO - Seleccione un archivo para análisis")

        if not _allowed_file(f.filename):
            return render_template("index.html", error="❌ FORMATO NO SOPORTADO - Use archivos CSV, Excel o JSON")

        # Procesamiento enterprise del archivo
        ext = os.path.splitext(f.filename)[1].lower()
        safe_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(_job_dir(job_id), safe_name)
        f.save(save_path)

        # Metadatos enterprise
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "payment_ok": payment_ok,
            "form": form,
            "upload_time": datetime.now().isoformat()
        }
        _save_meta(job_id, meta)

        # Previsualización enterprise - OPTIMIZADA
        try:
            df = parse_dataframe(save_path)
            device_profile = detect_device_profile(df, form.get("origen_app", ""))
            df = preprocess_data_universal(df, device_profile)
            
            profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
            
            # Generar tabla HTML enterprise (limitada)
            table_html = df.head(8).to_html(  # Solo 8 filas para previsualización
                classes="table table-striped table-bordered table-hover table-sm",
                index=False,
                escape=False
            )
            
            log.info(f"✅ Previsualización generada: {len(df)} registros | Dispositivo: {profile_name}")
            
            return render_template(
                "index.html",
                table_html=table_html,
                filename=f.filename,
                form_data=form,
                device_detected=profile_name,
                mensaje=mensaje,
                show_payment=(not payment_ok),
            )
            
        except Exception as e:
            log.error(f"❌ Error en procesamiento: {str(e)}")
            return render_template("index.html", error=f"❌ ERROR EN PROCESAMIENTO: {str(e)}")
            
    except Exception as e:
        log.error(f"💥 Error general en upload: {str(e)}")
        return render_template("index.html", error=f"❌ ERROR DEL SISTEMA: {str(e)}")

@app.route("/create_preference", methods=["POST"])
def create_preference():
    """Sistema de pago enterprise"""
    if not mp:
        return jsonify(error="SISTEMA DE PAGO NO CONFIGURADO"), 500

    job_id = session.get("job_id")
    if not job_id:
        return jsonify(error="SESIÓN INVÁLIDA"), 400

    # Precio enterprise por servicio premium universal
    price = 1000

    pref_data = {
        "items": [{
            "title": "InertiaX Enterprise - Análisis Universal Premium",
            "description": "Análisis biomecánico universal con IA científica para todos los dispositivos",
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
        log.error(f"Error en sistema de pago enterprise: {e}")
        return jsonify(error=str(e)), 500

@app.route("/success")
def success():
    """Pago exitoso - enterprise"""
    job_id = session.get("job_id")
    if not job_id:
        return redirect(url_for("index"))

    meta = _load_meta(job_id)
    meta["payment_ok"] = True
    _save_meta(job_id, meta)
    
    log.info(f"✅ Pago enterprise exitoso para job: {job_id}")
    return render_template("success.html")

@app.route("/cancel") 
def cancel():
    """Pago cancelado"""
    return render_template("cancel.html")

@app.route("/generate_report")
def generate_report():
    """Generación de reporte UNIVERSAL enterprise completo - OPTIMIZADO"""
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
        log.info("🚀 INICIANDO GENERACIÓN DE REPORTE ENTERPRISE OPTIMIZADO")
        
        # 1. Carga y procesamiento universal
        df = parse_dataframe(file_path)
        device_profile = detect_device_profile(df, meta.get("form", {}).get("origen_app", ""))
        df = preprocess_data_universal(df, device_profile)
        
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        log.info(f"📊 Dataset enterprise cargado: {df.shape[0]} registros | Dispositivo: {profile_name}")

        # 2. ANÁLISIS COMPLETO UNIVERSAL OPTIMIZADO
        log.info("🧠 EJECUTANDO ANÁLISIS UNIVERSAL OPTIMIZADO...")
        analysis_results = perform_comprehensive_analysis(df, device_profile)
        
        # 3. GENERACIÓN DE REPORTES ENTERPRISE OPTIMIZADOS
        log.info("📄 GENERANDO REPORTES OPTIMIZADOS...")
        pdf_path = generate_comprehensive_pdf(analysis_results, meta.get("form", {}))
        word_path = generate_word_report(analysis_results, meta.get("form", {}))
        
        # 4. CREACIÓN DE PAQUETE ENTERPRISE
        zip_path = os.path.join(_job_dir(job_id), f"reporte_enterprise_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "INERTIAX_ENTERPRISE_Reporte_Universal.pdf")
            if word_path and os.path.exists(word_path):
                zf.write(word_path, "INERTIAX_ENTERPRISE_Reporte_Universal.html")
            zf.write(file_path, f"datos_originales/{os.path.basename(meta.get('file_name', 'datos.csv'))}")
            
            # Agregar datos procesados
            processed_data_path = os.path.join(_job_dir(job_id), "datos_procesados.csv")
            df.to_csv(processed_data_path, index=False, encoding='utf-8')
            zf.write(processed_data_path, "datos_procesados/analisis_universal.csv")

        # Limpieza enterprise
        try:
            os.remove(pdf_path)
            if word_path and os.path.exists(word_path):
                os.remove(word_path)
            os.remove(processed_data_path)
        except:
            pass

        log.info("✅ REPORTE ENTERPRISE OPTIMIZADO GENERADO EXITOSAMENTE")
            
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"InertiaX_Enterprise_Reporte_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ ERROR EN GENERACIÓN DE REPORTE: {str(e)}")
        return render_template("index.html", error=f"❌ ERROR CRÍTICO: {str(e)}")

@app.route("/preview_analysis")
def preview_analysis():
    """Vista previa enterprise del análisis - SIMPLIFICADA"""
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
        device_profile = detect_device_profile(df, meta.get("form", {}).get("origen_app", ""))
        df = preprocess_data_universal(df, device_profile)
        
        # Solo análisis exploratorio para vista previa rápida
        exploratory_analysis = perform_exploratory_analysis(df, device_profile)
        session_report = generate_session_report(df, device_profile)
        
        return render_template(
            "preview.html",
            exploratory_analysis=exploratory_analysis,
            session_report=session_report,
            device_profile=device_profile,
            profile_name=app.config["DEVICE_PROFILES"][device_profile]["name"],
            filename=meta.get("file_name")
        )
        
    except Exception as e:
        log.error(f"Error en vista previa optimizada: {e}")
        return render_template("index.html", error=f"Error en vista previa: {e}")

# ==============================
# MANEJO DE ERRORES ENTERPRISE
# ==============================

@app.errorhandler(413)
def too_large(_e):
    return render_template("index.html", error="❌ ARCHIVO DEMASIADO GRANDE - Máximo 100MB")

@app.errorhandler(404)
def not_found(_e):
    return render_template("index.html", error="❌ RECURSO NO ENCONTRADO")

@app.errorhandler(500)
def internal_error(_e):
    return render_template("index.html", error="❌ ERROR INTERNO DEL SERVIDOR")

@app.errorhandler(Exception)
def global_error(e):
    log.exception("Error no controlado en sistema enterprise")
    return render_template("index.html", error=f"❌ ERROR DEL SISTEMA: {str(e)}")

# ==============================
# INICIALIZACIÓN ENTERPRISE
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX ENTERPRISE UNIVERSAL OPTIMIZADO INICIANDO EN PUERTO {port}")
    log.info(f"📱 DISPOSITIVOS SOPORTADOS: {list(app.config['DEVICE_PROFILES'].keys())}")
    app.run(host="0.0.0.0", port=port, debug=False)
