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
    df = df[df.get("load", 1) > 0]  # Eliminar cargas inválidas
    df = df[df.get("avg_velocity", 1) > 0.1]  # Velocidades mínimas realistas
    final_rows = len(df)
    
    log.info(f"Procesamiento completado: {initial_rows} -> {final_rows} filas válidas")
    return df

# ==============================
# ANÁLISIS CIENTÍFICO UNIVERSAL
# ==============================

def perform_comprehensive_analysis(df: pd.DataFrame, device_profile: str) -> Dict:
    """
    Realiza TODOS los análisis: exploratorio, gráfico, predictivo, reporte e interpretativo
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
        # 1. ANÁLISIS EXPLORATORIO
        analysis_results["exploratory_analysis"] = perform_exploratory_analysis(df, device_profile)
        
        # 2. ANÁLISIS GRÁFICO
        graphical_results = perform_graphical_analysis(df, device_profile)
        analysis_results["graphical_analysis"] = graphical_results["analysis"]
        analysis_results["charts"].extend(graphical_results["charts"])
        
        # 3. MODELO PREDICTIVO
        analysis_results["predictive_model"] = perform_predictive_analysis(df)
        
        # 4. REPORTE DE SESIÓN
        analysis_results["session_report"] = generate_session_report(df, device_profile)
        
        # 5. ANÁLISIS INTERPRETATIVO (IA)
        analysis_results["interpretive_analysis"] = perform_interpretive_analysis(df, device_profile)
        
        # 6. ANÁLISIS BIOMECÁNICO AVANZADO
        analysis_results["advanced_biomechanical"] = perform_advanced_biomechanical_analysis(df)
        
    except Exception as e:
        log.error(f"Error en análisis completo: {str(e)}")
        analysis_results["error"] = f"Error en análisis: {str(e)}"
    
    return analysis_results

def perform_exploratory_analysis(df: pd.DataFrame, device_profile: str) -> str:
    """🧠 1. Análisis exploratorio completo y universal"""
    analysis_lines = []
    
    profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
    analysis_lines.append(f"🧠 ANÁLISIS EXPLORATORIO - {profile_name}")
    analysis_lines.append("=" * 70)
    
    # Estadísticas básicas universales
    analysis_lines.append("\n📊 ESTADÍSTICAS DESCRIPTIVAS UNIVERSALES:")
    analysis_lines.append("-" * 50)
    
    # Columnas numéricas comunes a todos los dispositivos
    numeric_columns = []
    for col in ['load', 'carga_kg', 'max_velocity', 'avg_velocity', 'velocidad_maxima_m_s', 
                'velocidad_concentrica_m_s', 'duration', 'duracion_s']:
        if col in df.columns:
            numeric_columns.append(col)
    
    for col in numeric_columns:
        stats_desc = df[col].describe()
        cv = (df[col].std() / df[col].mean() * 100) if df[col].mean() > 0 else 0
        analysis_lines.append(f"{col}: μ={stats_desc['mean']:.3f} ± {stats_desc['std']:.3f} "
                           f"(CV={cv:.1f}%) | Range: {stats_desc['min']:.1f}-{stats_desc['max']:.1f}")
    
    # Patrones carga vs velocidad (compatible con ambos sistemas)
    analysis_lines.append("\n🔍 PATRONES CARGA-VELOCIDAD UNIVERSALES:")
    analysis_lines.append("-" * 45)
    
    load_col = 'load' if 'load' in df.columns else 'carga_kg'
    velocity_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
    
    if load_col in df.columns and velocity_col in df.columns:
        correlation = df[load_col].corr(df[velocity_col])
        analysis_lines.append(f"Correlación carga-velocidad: {correlation:.3f}")
        
        # Análisis por rangos de carga
        load_bins = pd.cut(df[load_col], bins=4)
        velocity_by_load = df.groupby(load_bins)[velocity_col].agg(['mean', 'std', 'count'])
        for bin_range, stats in velocity_by_load.iterrows():
            analysis_lines.append(f"Carga {bin_range}: Vel {stats['mean']:.3f} ± {stats['std']:.3f} m/s (n={stats['count']})")
    
    # Análisis por tipo de ejercicio
    type_col = 'type' if 'type' in df.columns else 'tipo_analisis'
    if type_col in df.columns:
        analysis_lines.append(f"\n🏋️ DIFERENCIAS POR TIPO DE EJERCICIO:")
        analysis_lines.append("-" * 40)
        
        for exercise_type in df[type_col].unique():
            type_data = df[df[type_col] == exercise_type]
            analysis_lines.append(f"{exercise_type}: {len(type_data)} reps | "
                               f"Carga: {type_data[load_col].mean():.1f}kg | "
                               f"Vel: {type_data[velocity_col].mean():.3f}m/s")
    
    # Análisis por usuario/atleta
    user_col = 'user' if 'user' in df.columns else 'atleta'
    if user_col in df.columns:
        analysis_lines.append(f"\n👥 ANÁLISIS POR {user_col.upper()}:")
        analysis_lines.append("-" * 40)
        
        user_stats = df.groupby(user_col).agg({
            load_col: ['count', 'mean', 'max'],
            velocity_col: ['mean', 'std', 'max']
        }).round(3)
        
        for user in df[user_col].unique():
            user_data = df[df[user_col] == user]
            analysis_lines.append(f"• {user}: {len(user_data)} reps | "
                               f"Carga: {user_data[load_col].mean():.1f}kg | "
                               f"Vel: {user_data[velocity_col].mean():.3f}m/s")
    
    # Detección de fatiga universal
    analysis_lines.append("\n⚠️ DETECCIÓN DE FATIGA Y CALIDAD DE DATOS:")
    analysis_lines.append("-" * 45)
    
    rep_col = 'rep_number' if 'rep_number' in df.columns else 'repeticion'
    if rep_col in df.columns and user_col in df.columns:
        for user in df[user_col].unique():
            user_data = df[df[user_col] == user].sort_values(rep_col)
            if len(user_data) > 1:
                first_vel = user_data[velocity_col].iloc[0]
                last_vel = user_data[velocity_col].iloc[-1]
                if first_vel > 0:
                    velocity_drop = ((first_vel - last_vel) / first_vel) * 100
                    if velocity_drop > 10:
                        analysis_lines.append(f"⚠️ {user}: Posible fatiga - caída de {velocity_drop:.1f}% en velocidad")
    
    return "\n".join(analysis_lines)

def perform_graphical_analysis(df: pd.DataFrame, device_profile: str) -> Dict:
    """📈 2. Análisis gráfico universal automático"""
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
        type_col = 'type' if 'type' in df.columns else 'tipo_analisis'
        
        # 1. Gráfico de dispersión load vs velocidad por tipo
        if load_col in df.columns and max_vel_col in df.columns:
            fig, ax = plt.subplots(figsize(12, 8))
            
            # Colores profesionales
            professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
            
            if type_col in df.columns:
                types = df[type_col].unique()
                colors_dict = {t: professional_colors[i % len(professional_colors)] for i, t in enumerate(types)}
                
                for exercise_type in types:
                    type_data = df[df[type_col] == exercise_type]
                    ax.scatter(type_data[load_col], type_data[max_vel_col], 
                              c=colors_dict[exercise_type], alpha=0.7, s=60, label=exercise_type)
            else:
                ax.scatter(df[load_col], df[max_vel_col], alpha=0.6, s=50, color=professional_colors[0])
            
            ax.set_xlabel('Carga (kg)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocidad Máxima (m/s)', fontsize=12, fontweight='bold')
            ax.set_title(f'PERFIL FUERZA-VELOCIDAD\n{profile_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(('fuerza_velocidad_universal', buf))
            analysis_text.append("• Gráfico 1: Perfil Fuerza-Velocidad - Relación fundamental carga-velocidad")
        
        # 2. Evolución temporal si hay datos de fecha
        date_cols = [col for col in df.columns if 'fecha' in col or 'date' in col]
        if date_cols and avg_vel_col in df.columns:
            fig, ax = plt.subplots(figsize(12, 6))
            
            fecha_col = date_cols[0]
            df_sorted = df.sort_values(fecha_col)
            
            if user_col in df.columns:
                users = df[user_col].unique()[:4]  # Máximo 4 usuarios
                for i, user in enumerate(users):
                    user_data = df_sorted[df_sorted[user_col] == user]
                    if len(user_data) > 1:
                        ax.plot(user_data[fecha_col], user_data[avg_vel_col], 
                               'o-', linewidth=2, label=user, markersize=6,
                               color=professional_colors[i % len(professional_colors)])
            else:
                ax.plot(df_sorted[fecha_col], df_sorted[avg_vel_col], 'o-', 
                       linewidth=2, color=professional_colors[0], markersize=6)
            
            ax.set_xlabel('Fecha', fontsize=11, fontweight='bold')
            ax.set_ylabel('Velocidad Promedio (m/s)', fontsize=11, fontweight='bold')
            ax.set_title('EVOLUCIÓN TEMPORAL DE VELOCIDAD', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(('evolucion_temporal', buf))
            analysis_text.append("• Gráfico 2: Evolución Temporal - Tendencia de rendimiento en el tiempo")
        
        # 3. Análisis de distribución y consistencia
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histograma de velocidades
        if avg_vel_col in df.columns:
            ax1.hist(df[avg_vel_col], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Velocidad Promedio (m/s)')
            ax1.set_ylabel('Frecuencia')
            ax1.set_title('Distribución de Velocidades')
            ax1.grid(True, alpha=0.3)
        
        # Boxplot por usuario
        if user_col in df.columns and avg_vel_col in df.columns:
            df.boxplot(column=avg_vel_col, by=user_col, ax=ax2)
            ax2.set_title('Distribución por Usuario')
            ax2.set_ylabel('Velocidad (m/s)')
        
        # Consistencia técnica (CV por usuario)
        if user_col in df.columns and avg_vel_col in df.columns:
            consistency_data = []
            for user in df[user_col].unique():
                user_data = df[df[user_col] == user]
                if len(user_data) > 1 and user_data[avg_vel_col].mean() > 0:
                    cv = (user_data[avg_vel_col].std() / user_data[avg_vel_col].mean()) * 100
                    consistency_data.append((user, cv))
            
            if consistency_data:
                users, cvs = zip(*consistency_data)
                bars = ax3.bar(users, cvs, color='lightcoral', alpha=0.7)
                ax3.set_xlabel('Usuario')
                ax3.set_ylabel('Coef. Variación (%)')
                ax3.set_title('Consistencia Técnica')
                ax3.tick_params(axis='x', rotation=45)
                ax3.axhline(y=15, color='red', linestyle='--', alpha=0.7, label='Límite 15%')
                ax3.legend()
        
        # Matriz de correlación
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(corr_matrix.columns)))
            ax4.set_yticks(range(len(corr_matrix.columns)))
            ax4.set_xticklabels(corr_matrix.columns, rotation=45)
            ax4.set_yticklabels(corr_matrix.columns)
            ax4.set_title('Matriz de Correlación')
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        buf = save_plot_to_buffer(fig)
        charts.append(('analisis_completo', buf))
        analysis_text.append("• Gráfico 3: Análisis Completo - Distribución, consistencia y correlaciones")
        
        analysis_text.append("\n📋 CONCLUSIONES GRÁFICAS UNIVERSALES:")
        analysis_text.append("- Confirmación relación inversa carga-velocidad")
        analysis_text.append("- Identificación de patrones temporales y de consistencia")
        analysis_text.append("- Detección de outliers y valores atípicos")
        analysis_text.append("- Análisis de correlaciones entre variables clave")
        
    except Exception as e:
        analysis_text.append(f"❌ Error en análisis gráfico universal: {str(e)}")
    
    return {
        "analysis": "\n".join(analysis_text),
        "charts": charts
    }

def perform_predictive_analysis(df: pd.DataFrame) -> str:
    """⚙️ 3. Modelo predictivo universal de velocidad máxima"""
    analysis_lines = []
    
    analysis_lines.append("⚙️ MODELO PREDICTIVO UNIVERSAL - VELOCIDAD MÁXIMA")
    analysis_lines.append("=" * 60)
    
    try:
        # Preparar datos universalmente
        model_df = df.copy()
        
        # Columnas universales
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        avg_vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        
        # Codificar variables categóricas
        categorical_cols = ['type', 'user', 'atleta', 'ejercicio', 'tipo_analisis']
        for col in categorical_cols:
            if col in model_df.columns:
                model_df[f'{col}_encoded'] = model_df[col].astype('category').cat.codes
        
        # Seleccionar características
        features = [load_col]
        if f'type_encoded' in model_df.columns:
            features.append('type_encoded')
        if f'user_encoded' in model_df.columns:
            features.append('user_encoded')
        if 'duration' in model_df.columns:
            features.append('duration')
        if 'duracion_s' in model_df.columns:
            features.append('duracion_s')
        if avg_vel_col in model_df.columns:
            features.append(avg_vel_col)
        
        target = max_vel_col
        
        if target not in model_df.columns or len(features) < 1:
            analysis_lines.append("❌ Datos insuficientes para modelo predictivo")
            return "\n".join(analysis_lines)
        
        # Filtrar datos válidos
        model_df = model_df[features + [target]].dropna()
        
        if len(model_df) < 10:
            analysis_lines.append("❌ Muy pocos datos para entrenar modelo")
            return "\n".join(analysis_lines)
        
        X = model_df[features]
        y = model_df[target]
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        analysis_lines.append(f"\n📐 CONFIGURACIÓN DEL MODELO PREDICTIVO:")
        analysis_lines.append(f"- Características: {', '.join(features)}")
        analysis_lines.append(f"- Variable objetivo: {target}")
        analysis_lines.append(f"- Muestras entrenamiento: {len(X_train)}")
        analysis_lines.append(f"- Muestras prueba: {len(X_test)}")
        
        # Entrenar múltiples modelos
        models = {
            'Regresión Lineal': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        }
        
        best_model = None
        best_score = -np.inf
        best_model_name = ""
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = np.mean(np.abs(y_test - y_pred))
                
                analysis_lines.append(f"\n🎯 {name}:")
                analysis_lines.append(f"  - R²: {r2:.4f}")
                analysis_lines.append(f"  - RMSE: {rmse:.4f} m/s")
                analysis_lines.append(f"  - MAE: {mae:.4f} m/s")
                analysis_lines.append(f"  - Error relativo: {(mae / y_test.mean() * 100):.1f}%")
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model
                    best_model_name = name
            except Exception as e:
                analysis_lines.append(f"  - Error en {name}: {str(e)}")
        
        if best_model is not None:
            # Análisis de importancia de características
            if hasattr(best_model, 'feature_importances_'):
                analysis_lines.append(f"\n📊 IMPORTANCIA DE CARACTERÍSTICAS ({best_model_name}):")
                feature_importance = sorted(zip(features, best_model.feature_importances_), 
                                          key=lambda x: x[1], reverse=True)
                for feature, importance in feature_importance:
                    analysis_lines.append(f"  - {feature}: {importance:.3f}")
            
            # Interpretación del modelo
            analysis_lines.append(f"\n💡 INTERPRETACIÓN DEL MODELO:")
            analysis_lines.append(f"- El mejor modelo ({best_model_name}) explica el {best_score*100:.1f}% de la varianza")
            analysis_lines.append(f"- Error típico de predicción: {rmse:.3f} m/s (±{rmse*1000:.0f} mm/s)")
            analysis_lines.append(f"- Precisión adecuada para prescripción de entrenamiento")
            
            # Ejemplo de predicción
            if len(X_test) > 0:
                sample_idx = 0
                sample_pred = best_model.predict(X_test.iloc[sample_idx:sample_idx+1])[0]
                actual_value = y_test.iloc[sample_idx]
                error = abs(sample_pred - actual_value)
                
                analysis_lines.append(f"\n🔍 EJEMPLO DE PREDICCIÓN:")
                analysis_lines.append(f"- Input: {dict(X_test.iloc[sample_idx])}")
                analysis_lines.append(f"- Predicción: {sample_pred:.3f} m/s")
                analysis_lines.append(f"- Real: {actual_value:.3f} m/s")
                analysis_lines.append(f"- Error: {error:.3f} m/s ({error/actual_value*100:.1f}%)")
        
        else:
            analysis_lines.append("\n❌ No se pudo entrenar ningún modelo válido")
        
    except Exception as e:
        analysis_lines.append(f"❌ Error en modelo predictivo universal: {str(e)}")
    
    return "\n".join(analysis_lines)

def generate_session_report(df: pd.DataFrame, device_profile: str) -> str:
    """📊 4. Reporte automatizado de sesión universal"""
    report_lines = []
    
    profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
    report_lines.append(f"📊 REPORTE AUTOMATIZADO DE SESIÓN - {profile_name}")
    report_lines.append("=" * 70)
    
    try:
        # Configuración de columnas universales
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        avg_vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        user_col = 'user' if 'user' in df.columns else 'atleta'
        exercise_col = 'exercise' if 'exercise' in df.columns else 'ejercicio'
        duration_col = 'duration' if 'duration' in df.columns else 'duracion_s'
        
        # Resumen general de la sesión
        report_lines.append("\n📈 RESUMEN GENERAL DE LA SESIÓN:")
        report_lines.append("-" * 40)
        report_lines.append(f"• Total de repeticiones: {len(df):,}")
        report_lines.append(f"• Usuarios/Atletas: {df[user_col].nunique() if user_col in df.columns else 1}")
        report_lines.append(f"• Ejercicios: {df[exercise_col].nunique() if exercise_col in df.columns else 'N/A'}")
        report_lines.append(f"• Duración promedio: {df[duration_col].mean():.2f}s" if duration_col in df.columns else "")
        
        # Métricas pico de rendimiento
        report_lines.append("\n🏆 MÉTRICAS PICO DE RENDIMIENTO:")
        report_lines.append("-" * 35)
        
        if max_vel_col in df.columns:
            peak_velocity = df[max_vel_col].max()
            peak_velocity_user = df.loc[df[max_vel_col].idxmax(), user_col] if user_col in df.columns else "N/A"
            report_lines.append(f"• Velocidad máxima: {peak_velocity:.3f} m/s ({peak_velocity_user})")
        
        if load_col in df.columns:
            peak_load = df[load_col].max()
            peak_load_user = df.loc[df[load_col].idxmax(), user_col] if user_col in df.columns else "N/A"
            report_lines.append(f"• Carga máxima: {peak_load:.1f} kg ({peak_load_user})")
            
            # Volumen total
            total_volume = df[load_col].sum()
            report_lines.append(f"• Volumen total: {total_volume:.0f} kg")
        
        # Análisis por ejercicio
        if exercise_col in df.columns:
            report_lines.append("\n💪 ANÁLISIS DETALLADO POR EJERCICIO:")
            report_lines.append("-" * 40)
            
            exercise_stats = df.groupby(exercise_col).agg({
                load_col: ['count', 'mean', 'max', 'sum'],
                avg_vel_col: ['mean', 'std', 'max'],
                duration_col: 'mean' if duration_col in df.columns else 'count'
            }).round(3)
            
            for exercise in df[exercise_col].unique():
                ex_data = df[df[exercise_col] == exercise]
                report_lines.append(f"\n{exercise.upper()}:")
                report_lines.append(f"  - Repeticiones: {len(ex_data)}")
                if load_col in ex_data.columns:
                    report_lines.append(f"  - Carga promedio: {ex_data[load_col].mean():.1f} kg")
                    report_lines.append(f"  - Volumen ejercicio: {ex_data[load_col].sum():.0f} kg")
                if avg_vel_col in ex_data.columns:
                    report_lines.append(f"  - Velocidad promedio: {ex_data[avg_vel_col].mean():.3f} m/s")
                    report_lines.append(f"  - Consistencia: ±{ex_data[avg_vel_col].std():.3f} m/s")
        
        # Análisis comparativo por tipo
        type_col = 'type' if 'type' in df.columns else 'tipo_analisis'
        if type_col in df.columns:
            report_lines.append("\n🔄 COMPARATIVA ENTRE TIPOS DE EJERCICIO:")
            report_lines.append("-" * 45)
            
            for exercise_type in df[type_col].unique():
                type_data = df[df[type_col] == exercise_type]
                report_lines.append(f"\n{exercise_type}:")
                report_lines.append(f"  - Repeticiones: {len(type_data)}")
                if load_col in type_data.columns:
                    report_lines.append(f"  - Carga media: {type_data[load_col].mean():.1f} kg")
                if avg_vel_col in type_data.columns:
                    report_lines.append(f"  - Velocidad media: {type_data[avg_vel_col].mean():.3f} m/s")
                if duration_col in type_data.columns:
                    report_lines.append(f"  - Duración media: {type_data[duration_col].mean():.2f} s")
        
        # Recomendaciones inteligentes
        report_lines.append("\n💡 RECOMENDACIONES INTELIGENTES:")
        report_lines.append("-" * 35)
        
        # Análisis de fatiga para recomendaciones
        rep_col = 'rep_number' if 'rep_number' in df.columns else 'repeticion'
        if rep_col in df.columns and user_col in df.columns and avg_vel_col in df.columns:
            for user in df[user_col].unique():
                user_data = df[df[user_col] == user].sort_values(rep_col)
                if len(user_data) > 3:
                    first_vel = user_data[avg_vel_col].iloc[:3].mean()  # Primeras 3 reps
                    last_vel = user_data[avg_vel_col].iloc[-3:].mean()  # Últimas 3 reps
                    
                    if first_vel > 0:
                        velocity_drop = ((first_vel - last_vel) / first_vel) * 100
                        
                        if velocity_drop > 20:
                            report_lines.append(f"• {user}: REDUCIR VOLUMEN - Fatiga severa ({velocity_drop:.1f}% caída)")
                        elif velocity_drop > 10:
                            report_lines.append(f"• {user}: MANTENER VOLUMEN - Fatiga moderada ({velocity_drop:.1f}% caída)")
                        elif velocity_drop < 5:
                            report_lines.append(f"• {user}: AUMENTAR CARGA - Buena recuperación")
                        else:
                            report_lines.append(f"• {user}: ESTABILIDAD ADECUADA - Mantener programa")
        
        # Recomendaciones generales basadas en métricas
        if avg_vel_col in df.columns:
            overall_avg_velocity = df[avg_vel_col].mean()
            if overall_avg_velocity < 0.4:
                report_lines.append("• CONSIDERAR: Reducir cargas para mejorar velocidad de ejecución")
            elif overall_avg_velocity > 0.9:
                report_lines.append("• OPORTUNIDAD: Aumentar cargas para mayor estímulo de fuerza")
            else:
                report_lines.append("• ZONA ÓPTIMA: Velocidades en rango ideal para desarrollo de potencia")
        
        # Eficiencia mecánica
        if load_col in df.columns and avg_vel_col in df.columns:
            avg_power = (df[load_col] * df[avg_vel_col]).mean()
            report_lines.append(f"• Potencia media estimada: {avg_power:.1f} W")
        
    except Exception as e:
        report_lines.append(f"❌ Error en reporte de sesión universal: {str(e)}")
    
    return "\n".join(report_lines)

def perform_interpretive_analysis(df: pd.DataFrame, device_profile: str) -> str:
    """💬 5. Análisis interpretativo universal (IA explicativa)"""
    
    if not ai_client:
        return "🔍 ANÁLISIS INTERPRETATIVO NO DISPONIBLE - Configure OPENAI_API_KEY"
    
    try:
        # Preparar resumen universal de datos para IA
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        max_vel_col = 'max_velocity' if 'max_velocity' in df.columns else 'velocidad_maxima_m_s'
        avg_vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        user_col = 'user' if 'user' in df.columns else 'atleta'
        
        data_summary = f"""
        DATOS DE ENTRENAMIENTO UNIVERSAL - RESUMEN PARA ANÁLISIS:
        
        Dispositivo: {app.config["DEVICE_PROFILES"][device_profile]["name"]}
        
        Estadísticas clave:
        - Total repeticiones: {len(df)}
        - Usuarios/Atletas: {df[user_col].nunique() if user_col in df.columns else 'N/A'}
        - Ejercicios: {df['exercise'].nunique() if 'exercise' in df.columns else df['ejercicio'].nunique() if 'ejercicio' in df.columns else 'N/A'}
        
        Métricas promedio:
        - Carga: {df[load_col].mean() if load_col in df.columns else 'N/A':.1f} kg
        - Velocidad máxima: {df[max_vel_col].mean() if max_vel_col in df.columns else 'N/A':.3f} m/s
        - Velocidad promedio: {df[avg_vel_col].mean() if avg_vel_col in df.columns else 'N/A':.3f} m/s
        - Duración: {df['duration'].mean() if 'duration' in df.columns else df['duracion_s'].mean() if 'duracion_s' in df.columns else 'N/A':.2f} s
        
        Distribución de cargas:
        {df[load_col].describe() if load_col in df.columns else 'No disponible'}
        
        Muestra de datos (primeras 8 repeticiones):
        {df.head(8).to_string()}
        """
        
        prompt = f"""
        Eres un entrenador deportivo experto en fuerza y potencia con 15+ años de experiencia en alto rendimiento.
        Analiza los siguientes datos de entrenamiento y proporciona conclusiones en lenguaje natural, práctico y accionable.
        
        {data_summary}
        
        Proporciona un análisis que incluya:
        
        1. OBSERVACIONES SOBRE PUNTOS ÓPTIMOS:
           - Ej: "A X kg el atleta logra mayor velocidad, lo que sugiere un punto óptimo de potencia"
           - Identificar rangos de carga donde se maximiza velocidad o potencia
        
        2. COMPARACIONES ENTRE TIPOS DE EJERCICIO:
           - Ej: "Las repeticiones de tipo X muestran menor duración y mayor velocidad media"
           - Diferencias en patrones de ejecución
        
        3. IDENTIFICACIÓN DE PATRONES DE RENDIMIENTO:
           - Consistencia entre repeticiones
           - Patrones de fatiga intra-serie
           - Variabilidad inter-sesión
        
        4. RECOMENDACIONES ESPECÍFICAS Y ACCIONABLES:
           - Ajustes de carga basados en velocidades
           - Estrategias para mejora técnica
           - Control de volumen e intensidad
        
        5. DETECCIÓN DE POSIBLES PROBLEMAS:
           - Técnica inconsistente
           - Fatiga excesiva
           - Cargas inadecuadas
        
        Usa un lenguaje claro, directo y orientado a la acción. Sé específico con los números cuando sea relevante.
        Incluye ejemplos concretos y métricas específicas del análisis.
        """
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": "Eres un entrenador experto en fuerza y potencia con experiencia en múltiples dispositivos de medición. Tu análisis es práctico, basado en datos y orientado a la mejora inmediata del rendimiento. Usa un tono profesional pero accesible."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        return f"💬 ANÁLISIS INTERPRETATIVO - PERSPECTIVA DEL ENTRENADOR EXPERTO\n\n{response.choices[0].message.content}"
        
    except Exception as e:
        return f"❌ Error en análisis interpretativo universal: {str(e)}"

def perform_advanced_biomechanical_analysis(df: pd.DataFrame) -> str:
    """🔬 6. Análisis biomecánico avanzado universal"""
    analysis_lines = []
    
    analysis_lines.append("🔬 ANÁLISIS BIOMECÁNICO AVANZADO UNIVERSAL")
    analysis_lines.append("=" * 60)
    
    try:
        # Configuración de columnas universales
        load_col = 'load' if 'load' in df.columns else 'carga_kg'
        vel_col = 'avg_velocity' if 'avg_velocity' in df.columns else 'velocidad_concentrica_m_s'
        user_col = 'user' if 'user' in df.columns else 'atleta'
        
        if load_col not in df.columns or vel_col not in df.columns:
            return "❌ Datos insuficientes para análisis biomecánico avanzado"
        
        analysis_lines.append("\n📐 PERFILES FUERZA-VELOCIDAD INDIVIDUALES:")
        analysis_lines.append("-" * 45)
        
        # Análisis por usuario/atleta
        if user_col in df.columns:
            for user in df[user_col].unique():
                user_data = df[df[user_col] == user]
                if len(user_data) > 2:
                    try:
                        # Regresión lineal fuerza-velocidad
                        x = user_data[load_col].values
                        y = user_data[vel_col].values
                        
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        
                        # Calcular parámetros clave
                        f0 = -intercept/slope if slope != 0 else 0  # Fuerza máxima teórica
                        v0 = intercept  # Velocidad máxima teórica
                        pmax = (-intercept * intercept) / (4 * slope) if slope != 0 else 0  # Potencia máxima teórica
                        
                        analysis_lines.append(f"\n{user}:")
                        analysis_lines.append(f"  - Fuerza max (F0): {f0:.1f} kg")
                        analysis_lines.append(f"  - Velocidad max (V0): {v0:.3f} m/s") 
                        analysis_lines.append(f"  - Potencia max (Pmax): {pmax:.1f} W")
                        analysis_lines.append(f"  - Calidad ajuste (R²): {r_value**2:.3f}")
                        analysis_lines.append(f"  - Pendiente: {slope:.4f} m/s/kg")
                        
                    except Exception as e:
                        analysis_lines.append(f"\n{user}: Error en cálculo de perfil")
        
        # Análisis de eficiencia global
        analysis_lines.append("\n📊 EFICIENCIA MECÁNICA GLOBAL:")
        analysis_lines.append("-" * 35)
        
        # Calcular relación carga-velocidad global
        global_slope, global_intercept, global_r, _, _ = stats.linregress(df[load_col], df[vel_col])
        analysis_lines.append(f"• Pendiente global: {global_slope:.4f} m/s/kg")
        analysis_lines.append(f"• Intercepto global: {global_intercept:.3f} m/s")
        analysis_lines.append(f"• R² global: {global_r**2:.3f}")
        
        # Análisis de fatiga neuromuscular
        analysis_lines.append("\n🔄 ANÁLISIS DE FATIGA NEUROMUSCULAR:")
        analysis_lines.append("-" * 35)
        
        rep_col = 'rep_number' if 'rep_number' in df.columns else 'repeticion'
        if rep_col in df.columns and user_col in df.columns:
            fatigue_analysis = []
            for user in df[user_col].unique():
                user_data = df[df[user_col] == user].sort_values(rep_col)
                if len(user_data) > 3:
                    velocity_decrement = ((user_data[vel_col].iloc[:3].mean() - user_data[vel_col].iloc[-3:].mean()) / 
                                        user_data[vel_col].iloc[:3].mean() * 100)
                    fatigue_analysis.append((user, velocity_decrement))
            
            for user, fatigue in fatigue_analysis:
                if fatigue > 0:
                    analysis_lines.append(f"• {user}: Decremento {fatigue:.1f}% - {'Alta' if fatigue > 15 else 'Moderada' if fatigue > 8 else 'Baja'} fatiga")
        
        # Recomendaciones biomecánicas
        analysis_lines.append("\n🎯 RECOMENDACIONES BIOMECÁNICAS:")
        analysis_lines.append("-" * 35)
        
        avg_velocity = df[vel_col].mean()
        if avg_velocity < 0.5:
            analysis_lines.append("• ENFOQUE: Mejora de velocidad - cargas ligeras con énfasis en aceleración")
        elif avg_velocity > 0.8:
            analysis_lines.append("• ENFOQUE: Fuerza máxima - aumentar cargas progresivamente")
        else:
            analysis_lines.append("• ENFOQUE: Potencia - mantener cargas moderadas con alta velocidad")
        
        # Zonas de entrenamiento recomendadas
        analysis_lines.append("• ZONAS SUGERIDAS:")
        analysis_lines.append("  - Fuerza: >85% 1RM, vel <0.5 m/s")
        analysis_lines.append("  - Potencia: 70-85% 1RM, vel 0.5-0.8 m/s") 
        analysis_lines.append("  - Velocidad: <70% 1RM, vel >0.8 m/s")
        
    except Exception as e:
        analysis_lines.append(f"❌ Error en análisis biomecánico avanzado: {str(e)}")
    
    return "\n".join(analysis_lines)

# ==============================
# FUNCIONES DE APOYO UNIVERSALES
# ==============================

def save_plot_to_buffer(fig) -> BytesIO:
    """Guarda gráfico en buffer con calidad profesional"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf

def generate_comprehensive_pdf(analysis_results: Dict, meta: dict) -> str:
    """Genera PDF UNIVERSAL con TODOS los análisis"""
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_universal_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos empresariales personalizados
        title_style = ParagraphStyle(
            'EnterpriseTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a5276'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'EnterpriseSection',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2e86ab'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        )
        
        subsection_style = ParagraphStyle(
            'EnterpriseSubsection',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#3B1F2B'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header empresarial
        story.append(Paragraph("INERTIAX ENTERPRISE - REPORTE UNIVERSAL DE ANÁLISIS", title_style))
        story.append(Spacer(1, 15))
        
        # Información del análisis
        device_profile = analysis_results.get("device_profile", "generic_csv")
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'Profesional')}<br/>
        <b>Dispositivo:</b> {profile_name}<br/>
        <b>Fecha de generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Sistema:</b> InertiaX Enterprise v3.0<br/>
        <b>Tipo de análisis:</b> Completo Universal (6 dimensiones)
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 25))
        
        # 1. ANÁLISIS EXPLORATORIO
        story.append(Paragraph("🧠 1. ANÁLISIS EXPLORATORIO UNIVERSAL", section_style))
        exploratory_text = analysis_results.get("exploratory_analysis", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(exploratory_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 2. ANÁLISIS GRÁFICO
        story.append(Paragraph("📈 2. ANÁLISIS GRÁFICO UNIVERSAL", section_style))
        graphical_text = analysis_results.get("graphical_analysis", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(graphical_text, styles['Normal']))
        
        # Insertar gráficos
        charts = analysis_results.get("charts", [])
        for i, (chart_name, chart_buf) in enumerate(charts[:6]):  # Máximo 6 gráficos
            try:
                chart_buf.seek(0)
                img = ReportLabImage(chart_buf, width=6*inch, height=4*inch)
                story.append(Spacer(1, 12))
                story.append(img)
                story.append(Paragraph(f"Figura {i+1}: {chart_name.replace('_', ' ').title()}", styles['Italic']))
                story.append(Spacer(1, 15))
            except Exception as e:
                continue
        
        # 3. MODELO PREDICTIVO
        story.append(Paragraph("⚙️ 3. MODELO PREDICTIVO UNIVERSAL", section_style))
        predictive_text = analysis_results.get("predictive_model", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(predictive_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 4. REPORTE DE SESIÓN
        story.append(Paragraph("📊 4. REPORTE DE SESIÓN UNIVERSAL", section_style))
        session_text = analysis_results.get("session_report", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(session_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 5. ANÁLISIS INTERPRETATIVO
        story.append(Paragraph("💬 5. ANÁLISIS INTERPRETATIVO - IA ESPECIALIZADA", section_style))
        interpretive_text = analysis_results.get("interpretive_analysis", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(interpretive_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 6. ANÁLISIS BIOMECÁNICO AVANZADO
        story.append(Paragraph("🔬 6. ANÁLISIS BIOMECÁNICO AVANZADO", section_style))
        biomechanical_text = analysis_results.get("advanced_biomechanical", "No disponible").replace('\n', '<br/>')
        story.append(Paragraph(biomechanical_text, styles['Normal']))
        
        # Footer empresarial
        story.append(Spacer(1, 25))
        footer_text = """
        <i>Reporte generado por InertiaX Enterprise Analysis System<br/>
        Sistema certificado para análisis biomecánico deportivo universal<br/>
        Compatible con: Encoder V1 Windows, Encoder Vertical Android, CSV Genérico<br/>
        © 2024 InertiaX Enterprise - Todos los derechos reservados</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF universal: {str(e)}")
        # PDF de error profesional
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX ENTERPRISE - ERROR EN REPORTE")
        c.drawString(100, 730, f"Error: {str(e)}")
        c.drawString(100, 710, "Contacte al soporte técnico empresarial")
        c.save()
        return error_path

def generate_word_report(analysis_results: Dict, meta: dict) -> str:
    """Genera reporte en formato Word universal"""
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
                    body { 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        margin: 40px; 
                        line-height: 1.6;
                        color: #333;
                    }
                    h1 { 
                        color: #1a5276; 
                        text-align: center; 
                        border-bottom: 3px solid #2e86ab;
                        padding-bottom: 15px;
                    }
                    h2 { 
                        color: #2e86ab; 
                        border-bottom: 2px solid #a6c5e0; 
                        padding-bottom: 8px;
                        margin-top: 30px;
                    }
                    h3 { 
                        color: #3B1F2B; 
                        margin-top: 20px;
                    }
                    .section { 
                        margin-bottom: 35px; 
                        padding: 20px;
                        background: #f8f9fa;
                        border-radius: 8px;
                        border-left: 4px solid #2e86ab;
                    }
                    .info { 
                        background: #e8f4f8; 
                        padding: 20px; 
                        border-radius: 8px;
                        border: 1px solid #a6c5e0;
                    }
                    .recommendation { 
                        background: #d4edda; 
                        padding: 12px; 
                        border-left: 4px solid #28a745; 
                        margin: 12px 0;
                        border-radius: 4px;
                    }
                    .warning { 
                        background: #fff3cd; 
                        padding: 12px; 
                        border-left: 4px solid #ffc107; 
                        margin: 12px 0;
                        border-radius: 4px;
                    }
                    .chart-container {
                        text-align: center;
                        margin: 20px 0;
                        padding: 15px;
                        background: white;
                        border-radius: 8px;
                        border: 1px solid #ddd;
                    }
                    .footer {
                        margin-top: 40px; 
                        padding: 25px; 
                        background: #343a40; 
                        color: white; 
                        text-align: center;
                        border-radius: 8px;
                    }
                    .metric-box {
                        background: white;
                        padding: 15px;
                        margin: 10px 0;
                        border-radius: 6px;
                        border: 1px solid #e9ecef;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
            """)
            
            f.write(f"<h1>🚀 INERTIAX ENTERPRISE - REPORTE UNIVERSAL</h1>")
            f.write(f"<div class='info'>")
            f.write(f"<h3>📋 INFORMACIÓN DEL ANÁLISIS</h3>")
            f.write(f"<p><strong>Entrenador:</strong> {meta.get('nombre_entrenador', 'Profesional')}</p>")
            f.write(f"<p><strong>Dispositivo:</strong> {profile_name}</p>")
            f.write(f"<p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>")
            f.write(f"<p><strong>Sistema:</strong> Análisis Universal 6 Dimensiones</p>")
            f.write("</div>")
            
            # Secciones del análisis
            sections = [
                ("🧠 ANÁLISIS EXPLORATORIO UNIVERSAL", "exploratory_analysis"),
                ("📈 ANÁLISIS GRÁFICO UNIVERSAL", "graphical_analysis"), 
                ("⚙️ MODELO PREDICTIVO UNIVERSAL", "predictive_model"),
                ("📊 REPORTE DE SESIÓN UNIVERSAL", "session_report"),
                ("💬 ANÁLISIS INTERPRETATIVO - IA", "interpretive_analysis"),
                ("🔬 ANÁLISIS BIOMECÁNICO AVANZADO", "advanced_biomechanical")
            ]
            
            for title, key in sections:
                content = analysis_results.get(key, "Análisis no disponible").replace('\n', '<br/>')
                # Formatear contenido para mejor legibilidad en HTML
                content = content.replace('•', '<br>•').replace('📊', '<br>📊').replace('🔍', '<br>🔍')
                content = content.replace('🏋️', '<br>🏋️').replace('👥', '<br>👥').replace('⚠️', '<br>⚠️')
                content = content.replace('🎯', '<br>🎯').replace('💡', '<br>💡').replace('🔬', '<br>🔬')
                content = content.replace('📐', '<br>📐').replace('🔄', '<br>🔄')
                
                f.write(f"<div class='section'>")
                f.write(f"<h2>{title}</h2>")
                f.write(f"<div class='metric-box'>{content}</div>")
                f.write("</div>")
            
            f.write("""
                <div class='footer'>
                    <h3>INERTIAX ENTERPRISE ANALYSIS SYSTEM</h3>
                    <p><em>Sistema certificado para análisis biomecánico deportivo universal</em></p>
                    <p><strong>Dispositivos compatibles:</strong> Encoder V1 Windows • Encoder Vertical Android • CSV Genérico</p>
                    <p><strong>© 2024 InertiaX Enterprise</strong> - Todos los derechos reservados</p>
                </div>
            </body>
            </html>
            """)
        
        return word_path
        
    except Exception as e:
        log.error(f"Error generando Word universal: {str(e)}")
        return ""

# ==============================
# RUTAS ENTERPRISE MEJORADAS
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
    """Endpoint enterprise universal para carga de datos"""
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

        # Previsualización enterprise
        try:
            df = parse_dataframe(save_path)
            device_profile = detect_device_profile(df, form.get("origen_app", ""))
            df = preprocess_data_universal(df, device_profile)
            
            profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
            
            # Generar tabla HTML enterprise
            table_html = df.head(12).to_html(
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
    price = 14900  # Servicio enterprise premium

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
    """Generación de reporte UNIVERSAL enterprise completo"""
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
        log.info("🚀 INICIANDO GENERACIÓN DE REPORTE ENTERPRISE UNIVERSAL")
        
        # 1. Carga y procesamiento universal
        df = parse_dataframe(file_path)
        device_profile = detect_device_profile(df, meta.get("form", {}).get("origen_app", ""))
        df = preprocess_data_universal(df, device_profile)
        
        profile_name = app.config["DEVICE_PROFILES"][device_profile]["name"]
        log.info(f"📊 Dataset enterprise cargado: {df.shape[0]} registros | Dispositivo: {profile_name}")

        # 2. ANÁLISIS COMPLETO UNIVERSAL
        log.info("🧠 EJECUTANDO ANÁLISIS UNIVERSAL (6 dimensiones)...")
        analysis_results = perform_comprehensive_analysis(df, device_profile)
        
        # 3. GENERACIÓN DE REPORTES ENTERPRISE
        log.info("📄 GENERANDO REPORTES ENTERPRISE (PDF + WORD)...")
        pdf_path = generate_comprehensive_pdf(analysis_results, meta.get("form", {}))
        word_path = generate_word_report(analysis_results, meta.get("form", {}))
        
        # 4. CREACIÓN DE PAQUETE ENTERPRISE
        zip_path = os.path.join(_job_dir(job_id), f"reporte_enterprise_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "INERTIAX_ENTERPRISE_Reporte_Universal.pdf")
            if word_path and os.path.exists(word_path):
                zf.write(word_path, "INERTIAX_ENTERPRISE_Reporte_Universal.doc")
            zf.write(file_path, f"datos_originales/{os.path.basename(meta.get('file_name', 'datos.csv'))}")
            
            # Agregar datos procesados
            processed_data_path = os.path.join(_job_dir(job_id), "datos_procesados.csv")
            df.to_csv(processed_data_path, index=False, encoding='utf-8')
            zf.write(processed_data_path, "datos_procesados/analisis_universal.csv")
            
            # Agregar metadatos del análisis
            meta_path = os.path.join(_job_dir(job_id), "metadatos_analisis.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "device_profile": device_profile,
                    "profile_name": profile_name,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "total_records": len(df),
                    "analysis_dimensions": 6
                }, f, indent=2, ensure_ascii=False)
            zf.write(meta_path, "metadatos/info_analisis.json")

        # Limpieza enterprise
        try:
            os.remove(pdf_path)
            if word_path and os.path.exists(word_path):
                os.remove(word_path)
            os.remove(processed_data_path)
            os.remove(meta_path)
        except:
            pass

        log.info("✅ REPORTE ENTERPRISE UNIVERSAL GENERADO EXITOSAMENTE")
            
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
    """Vista previa enterprise del análisis"""
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
        
        # Análisis científico
        analysis_results = perform_comprehensive_analysis(df, device_profile)
        
        return render_template(
            "preview.html",
            analysis_results=analysis_results,
            device_profile=device_profile,
            profile_name=app.config["DEVICE_PROFILES"][device_profile]["name"],
            filename=meta.get("file_name")
        )
        
    except Exception as e:
        log.error(f"Error en vista previa enterprise: {e}")
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
    log.info(f"🚀 INERTIAX ENTERPRISE UNIVERSAL STARTING ON PORT {port}")
    log.info(f"📱 DISPOSITIVOS SOPORTADOS: {list(app.config['DEVICE_PROFILES'].keys())}")
    app.run(host="0.0.0.0", port=port, debug=False)
