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

def preprocess_data_by_origin(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    """
    Procesamiento científico de datos para encoder vertical
    """
    log.info(f"Iniciando procesamiento científico para: {origin}")
    origin = origin.lower()
    
    if origin == "app_android_encoder_vertical":
        log.info("Procesando datos de Encoder Vertical Android")
        
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
        df = df[df["carga_kg"] > 0]  # Eliminar cargas inválidas
        final_rows = len(df)
        
        log.info(f"Procesamiento completado: {initial_rows} -> {final_rows} filas válidas")
        return df

    else:
        # Standardización profesional
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df

# ==============================
# ANÁLISIS CIENTÍFICO AVANZADO
# ==============================

def generate_comprehensive_stats(df: pd.DataFrame) -> str:
    """Genera estadísticas científicas completas"""
    stats_lines = []
    
    # Estadísticas generales del dataset
    stats_lines.append("📊 ANÁLISIS ESTADÍSTICO COMPLETO DEL DATASET")
    stats_lines.append("=" * 60)
    stats_lines.append(f"• Total de registros: {df.shape[0]:,}")
    stats_lines.append(f"• Total de variables: {df.shape[1]}")
    stats_lines.append(f"• Atletas únicos: {df['atleta'].nunique() if 'atleta' in df.columns else 'N/A'}")
    stats_lines.append(f"• Ejercicios únicos: {df['ejercicio'].nunique() if 'ejercicio' in df.columns else 'N/A'}")
    
    # Análisis temporal avanzado
    if 'fecha' in df.columns:
        stats_lines.append("\n📅 ANÁLISIS TEMPORAL")
        stats_lines.append("-" * 40)
        stats_lines.append(f"• Período: {df['fecha'].min().strftime('%Y-%m-%d')} a {df['fecha'].max().strftime('%Y-%m-%d')}")
        stats_lines.append(f"• Días de entrenamiento: {df['fecha'].nunique()}")
        stats_lines.append(f"• Sesiones por día: {df.groupby('fecha').size().mean():.1f}")
    
    # Métricas biomecánicas clave
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats_lines.append("\n🔬 MÉTRICAS BIOMECÁNICAS PRINCIPALES")
        stats_lines.append("-" * 40)
        
        key_metrics = ['carga_kg', 'velocidad_concentrica_m_s', 'velocidad_maxima_m_s', 'estimado_1rm_kg']
        for metric in key_metrics:
            if metric in df.columns:
                desc = df[metric].describe()
                cv = (df[metric].std() / df[metric].mean() * 100) if df[metric].mean() > 0 else 0
                stats_lines.append(
                    f"• {metric.replace('_', ' ').title()}: "
                    f"μ={desc['mean']:.2f} ± {desc['std']:.2f} "
                    f"(CV={cv:.1f}%) | "
                    f"Range: {desc['min']:.1f}-{desc['max']:.1f}"
                )
    
    # Análisis por atleta
    if 'atleta' in df.columns:
        stats_lines.append("\n👥 ANÁLISIS POR ATLETA")
        stats_lines.append("-" * 40)
        athlete_stats = df.groupby('atleta').agg({
            'carga_kg': ['count', 'mean', 'max'],
            'velocidad_concentrica_m_s': ['mean', 'std']
        }).round(3)
        
        for athlete in df['atleta'].unique():
            athlete_data = df[df['atleta'] == athlete]
            stats_lines.append(f"• {athlete}: {len(athlete_data)} reps | "
                             f"Carga: {athlete_data['carga_kg'].mean():.1f}kg | "
                             f"Vel: {athlete_data['velocidad_concentrica_m_s'].mean():.3f}m/s")
    
    return "\n".join(stats_lines)

def perform_advanced_biomechanical_analysis(df: pd.DataFrame) -> Dict:
    """Análisis biomecánico científico avanzado"""
    analysis = {
        "athlete_profiles": {},
        "exercise_analysis": {},
        "fatigue_analysis": {},
        "performance_metrics": {},
        "technical_consistency": {}
    }
    
    if 'atleta' not in df.columns:
        return analysis
    
    # Análisis individual por atleta
    for athlete in df['atleta'].unique():
        athlete_data = df[df['atleta'] == athlete]
        profile = {
            "total_volume": len(athlete_data),
            "avg_load": athlete_data['carga_kg'].mean() if 'carga_kg' in athlete_data else 0,
            "avg_velocity": athlete_data['velocidad_concentrica_m_s'].mean() if 'velocidad_concentrica_m_s' in athlete_data else 0,
            "max_velocity": athlete_data['velocidad_maxima_m_s'].max() if 'velocidad_maxima_m_s' in athlete_data else 0,
            "estimated_1rm": athlete_data['estimado_1rm_kg'].max() if 'estimado_1rm_kg' in athlete_data else 0,
            "velocity_decrement": calculate_velocity_decrement(athlete_data),
            "power_output": calculate_power_metrics(athlete_data)
        }
        analysis["athlete_profiles"][athlete] = profile
    
    # Análisis de fatiga avanzado
    if 'repeticion' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
        analysis["fatigue_analysis"] = analyze_fatigue_patterns(df)
    
    # Análisis de consistencia técnica
    analysis["technical_consistency"] = analyze_technical_consistency(df)
    
    return analysis

def calculate_velocity_decrement(athlete_data: pd.DataFrame) -> float:
    """Calcula el decremento de velocidad intra-serie"""
    if 'repeticion' not in athlete_data.columns or 'velocidad_concentrica_m_s' not in athlete_data.columns:
        return 0
    
    try:
        first_rep = athlete_data[athlete_data['repeticion'] == athlete_data['repeticion'].min()]['velocidad_concentrica_m_s'].mean()
        last_rep = athlete_data[athlete_data['repeticion'] == athlete_data['repeticion'].max()]['velocidad_concentrica_m_s'].mean()
        
        if first_rep > 0:
            return ((first_rep - last_rep) / first_rep) * 100
        return 0
    except:
        return 0

def calculate_power_metrics(athlete_data: pd.DataFrame) -> Dict:
    """Calcula métricas de potencia avanzadas"""
    metrics = {
        "avg_power": 0,
        "peak_power": 0,
        "power_endurance": 0
    }
    
    if 'potencia_w' in athlete_data.columns:
        metrics["avg_power"] = athlete_data['potencia_w'].mean()
        metrics["peak_power"] = athlete_data['potencia_w'].max()
    
    return metrics

def analyze_fatigue_patterns(df: pd.DataFrame) -> Dict:
    """Análisis científico de patrones de fatiga"""
    fatigue_analysis = {}
    
    if 'repeticion' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
        for athlete in df['atleta'].unique():
            athlete_data = df[df['atleta'] == athlete]
            fatigue_rates = []
            
            for exercise in athlete_data['ejercicio'].unique() if 'ejercicio' in athlete_data else ['default']:
                ex_data = athlete_data[athlete_data['ejercicio'] == exercise]
                if len(ex_data) > 1:
                    velocity_decrement = calculate_velocity_decrement(ex_data)
                    fatigue_rates.append(velocity_decrement)
            
            fatigue_analysis[athlete] = {
                "avg_fatigue_rate": np.mean(fatigue_rates) if fatigue_rates else 0,
                "fatigue_consistency": np.std(fatigue_rates) if fatigue_rates else 0
            }
    
    return fatigue_analysis

def analyze_technical_consistency(df: pd.DataFrame) -> Dict:
    """Análisis de consistencia técnica"""
    consistency = {}
    
    if 'velocidad_concentrica_m_s' in df.columns:
        for athlete in df['atleta'].unique():
            athlete_data = df[df['atleta'] == athlete]
            velocity_cv = (athlete_data['velocidad_concentrica_m_s'].std() / 
                          athlete_data['velocidad_concentrica_m_s'].mean() * 100)
            
            consistency[athlete] = {
                "velocity_cv": velocity_cv,
                "consistency_level": "Excelente" if velocity_cv < 10 else 
                                   "Buena" if velocity_cv < 20 else 
                                   "Moderada" if velocity_cv < 30 else "Baja"
            }
    
    return consistency

# ==============================
# IA PROFESIONAL - ANÁLISIS CIENTÍFICO
# ==============================

def run_professional_ai_analysis(df: pd.DataFrame, meta: dict) -> dict:
    """
    Análisis científico profesional por IA especializada
    """
    if not ai_client:
        return {
            "analysis": "🔬 SERVICIO DE IA NO DISPONIBLE - Configure OPENAI_API_KEY",
            "python_code_for_charts": "",
            "charts_description": "",
            "recommendations": ["Configurar API key de OpenAI para análisis profesional"]
        }
    
    try:
        # Análisis científico avanzado
        biomechanical_analysis = perform_advanced_biomechanical_analysis(df)
        comprehensive_stats = generate_comprehensive_stats(df)
        
        # Preparar datos completos para IA
        data_completa = df.to_csv(index=False)
        
        # Contexto científico profesional
        contexto = f"""
ANÁLISIS CIENTÍFICO PROFESIONAL - SISTEMA INERTIAX PRO
===================================================

INFORMACIÓN DEL ANÁLISIS:
• Entrenador: {meta.get('nombre_entrenador', 'Profesional del Deporte')}
• Origen de datos: Encoder Vertical Android (Sistema Profesional)
• Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
• Total de datos procesados: {df.shape[0]:,} registros

ESTADÍSTICAS CIENTÍFICAS COMPLETAS:
{comprehensive_stats}

ANÁLISIS BIOMECÁNICO AVANZADO:
• Perfiles individuales de {len(biomechanical_analysis.get('athlete_profiles', {}))} atletas
• Métricas de fatiga y consistencia técnica calculadas
• Análisis de potencia y eficiencia mecánica

INSTRUCCIONES PARA ANÁLISIS CIENTÍFICO PROFESIONAL:

REQUERIMIENTOS DE ANÁLISIS:

1. EVALUACIÓN BIOMECÁNICA COMPLETA:
   - Análisis individualizado por atleta con métricas específicas
   - Perfiles fuerza-velocidad con regresiones lineales
   - Eficiencia mecánica y técnica de ejecución
   - Identificación de asimetrías y desbalances

2. ANÁLISIS DE RENDIMIENTO AVANZADO:
   - Velocidad media, pico y decremento intra-serie
   - Potencia mecánica y producción de fuerza
   - Fatiga neuromuscular y recuperación
   - Capacidad de trabajo y tolerancia a la carga

3. EVALUACIÓN TÉCNICA Y CONSISTENCIA:
   - Variabilidad inter-repetición e inter-sesión
   - Consistencia en la ejecución técnica
   - Patrones de fatiga y mantenimiento técnico
   - Eficiencia del movimiento

4. RECOMENDACIONES CIENTÍFICAS:
   - Prescripción de carga basada en velocidad
   - Optimización de volumen e intensidad
   - Estrategias para mejora técnica
   - Prevención de sobreentrenamiento y lesiones

5. PROTOCOLO DE GRÁFICOS PROFESIONALES:
   - Gráficos de dispersión fuerza-velocidad
   - Curvas de fatiga y rendimiento
   - Evolución temporal de métricas clave
   - Comparativas entre atletas y ejercicios

DATOS COMPLETOS PARA ANÁLISIS CIENTÍFICO:
"""
        
        system_prompt = """
Eres un equipo de científicos deportivos con PhD en Biomecánica, Fisiología del Ejercicio y Analítica Deportiva. 
Tienes 20+ años de experiencia en alto rendimiento y investigación científica.

PROTOCOLO DE ANÁLISIS CIENTÍFICO:

1. ANÁLISIS BIOMECÁNICO PROFUNDO:
   - Evaluar perfiles individuales fuerza-velocidad con análisis de regresión
   - Calcular eficiencia mecánica y coeficientes de rendimiento
   - Identificar patrones de fatiga neuromuscular
   - Analizar consistencia técnica y variabilidad

2. METODOLOGÍA ESTADÍSTICA:
   - Utilizar análisis descriptivos y inferenciales
   - Calcular coeficientes de variación y desviaciones
   - Aplicar tests de normalidad y significancia
   - Generar intervalos de confianza

3. INTERPRETACIÓN CIENTÍFICA:
   - Basar conclusiones en evidencia estadística
   - Considerar contexto del entrenamiento
   - Identificar limitaciones y sesgos
   - Proponer hipótesis verificables

4. COMUNICACIÓN PROFESIONAL:
   - Usar terminología científica apropiada
   - Presentar datos con precisión y claridad
   - Incluir implicaciones prácticas
   - Proporcionar fundamento científico

RESPONDER EN FORMATO JSON ESTRICTAMENTE:
{
    "analysis": "Análisis científico completo...",
    "python_code_for_charts": "Código Python para gráficos profesionales...",
    "charts_description": "Descripción detallada de visualizaciones...", 
    "recommendations": [
        "Recomendación 1 basada en evidencia científica...",
        "Recomendación 2 con fundamento biomecánico..."
    ]
}

El análisis debe ser exhaustivo, científico y accionable.
"""

        user_prompt = f"{contexto}\n```csv\n{data_completa}\n```"

        log.info("🧠 INICIANDO ANÁLISIS CIENTÍFICO CON IA...")
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Máxima precisión científica
            max_tokens=8000,  # Análisis exhaustivo
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

# ==============================
# VISUALIZACIONES PROFESIONALES
# ==============================

def generate_professional_charts(df: pd.DataFrame, analysis: Dict) -> List[BytesIO]:
    """
    Genera visualizaciones científicas profesionales
    """
    charts = []
    plt.style.use('seaborn-v0_8-whitegrid')
    
    try:
        # Configuración profesional
        professional_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
        
        # 1. Gráfico de Perfiles Fuerza-Velocidad
        if 'atleta' in df.columns and 'carga_kg' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            for i, athlete in enumerate(df['atleta'].unique()):
                athlete_data = df[df['atleta'] == athlete]
                color = professional_colors[i % len(professional_colors)]
                
                # Gráfico de dispersión
                scatter = ax.scatter(athlete_data['carga_kg'], athlete_data['velocidad_concentrica_m_s'],
                                   c=color, alpha=0.7, s=60, label=athlete, edgecolors='white', linewidth=0.5)
                
                # Línea de tendencia
                if len(athlete_data) > 1:
                    z = np.polyfit(athlete_data['carga_kg'], athlete_data['velocidad_concentrica_m_s'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(athlete_data['carga_kg'].min(), athlete_data['carga_kg'].max(), 100)
                    ax.plot(x_range, p(x_range), color=color, linestyle='--', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Carga (kg)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Velocidad Concéntrica (m/s)', fontsize=12, fontweight='bold')
            ax.set_title('PERFIL FUERZA-VELOCIDAD POR ATLETA\nAnálisis Biomecánico Profesional', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Añadir anotaciones profesionales
            ax.text(0.02, 0.98, 'Análisis InertiaX Pro', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(buf)

        # 2. Evolución de Velocidad por Repetición
        if 'repeticion' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            athletes_to_show = df['atleta'].unique()[:4]  # Máximo 4 para claridad
            for i, athlete in enumerate(athletes_to_show):
                athlete_data = df[df['atleta'] == athlete]
                velocity_by_rep = athlete_data.groupby('repeticion')['velocidad_concentrica_m_s'].mean()
                
                ax.plot(velocity_by_rep.index, velocity_by_rep.values, 
                       marker='o', linewidth=2.5, markersize=6, 
                       color=professional_colors[i], label=athlete)
            
            ax.set_xlabel('Número de Repetición', fontsize=11, fontweight='bold')
            ax.set_ylabel('Velocidad Concéntrica (m/s)', fontsize=11, fontweight='bold')
            ax.set_title('EVOLUCIÓN DE VELOCIDAD POR REPETICIÓN\nAnálisis de Fatiga Intra-Serie', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(buf)

        # 3. Comparativa de Rendimiento entre Atletas
        if 'atleta' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Velocidad promedio
            avg_velocity = df.groupby('atleta')['velocidad_concentrica_m_s'].mean().sort_values(ascending=False)
            bars1 = ax1.bar(avg_velocity.index, avg_velocity.values, 
                           color=professional_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax1.set_title('VELOCIDAD MEDIA POR ATLETA', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Velocidad (m/s)', fontsize=11)
            ax1.tick_params(axis='x', rotation=45)
            
            # Añadir valores en barras
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Carga promedio
            if 'carga_kg' in df.columns:
                avg_load = df.groupby('atleta')['carga_kg'].mean().sort_values(ascending=False)
                bars2 = ax2.bar(avg_load.index, avg_load.values,
                               color=professional_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                
                ax2.set_title('CARGA MEDIA POR ATLETA', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Carga (kg)', fontsize=11)
                ax2.tick_params(axis='x', rotation=45)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(buf)

        # 4. Análisis de Consistencia Técnica
        if 'atleta' in df.columns and 'velocidad_concentrica_m_s' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            consistency_data = []
            for athlete in df['atleta'].unique():
                athlete_data = df[df['atleta'] == athlete]
                cv = (athlete_data['velocidad_concentrica_m_s'].std() / 
                      athlete_data['velocidad_concentrica_m_s'].mean() * 100)
                consistency_data.append((athlete, cv))
            
            athletes, cvs = zip(*sorted(consistency_data, key=lambda x: x[1]))
            bars = ax.bar(athletes, cvs, color=professional_colors, alpha=0.7)
            
            ax.set_xlabel('Atleta', fontsize=11, fontweight='bold')
            ax.set_ylabel('Coeficiente de Variación (%)', fontsize=11, fontweight='bold')
            ax.set_title('CONSISTENCIA TÉCNICA - ANÁLISIS DE VARIABILIDAD', 
                        fontsize=13, fontweight='bold', pad=15)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Líneas de referencia para consistencia
            ax.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Excelente (<10%)')
            ax.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Buena (<20%)')
            ax.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Moderada (<30%)')
            ax.legend()
            
            plt.tight_layout()
            buf = save_plot_to_buffer(fig)
            charts.append(buf)
            
    except Exception as e:
        log.error(f"Error en generación de gráficos profesionales: {str(e)}")
    
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

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
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
        
        # Información del análisis
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'Profesional')}<br/>
        <b>Fecha de generación:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>
        <b>Sistema:</b> InertiaX Professional v2.0<br/>
        <b>Tipo de análisis:</b> Biomecánico Deportivo Avanzado
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
            
            for i, chart in enumerate(charts[:6]):
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 5))
                    story.append(Paragraph(f"Figura {i+1}: {ai_result.get('charts_description', 'Gráfico profesional').split('.')[0] if i == 0 else 'Análisis continuado'}", styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Recomendaciones profesionales
        story.append(Paragraph("RECOMENDACIONES CIENTÍFICAS", subtitle_style))
        recommendations = ai_result.get('recommendations', [])
        if isinstance(recommendations, list):
            for rec in recommendations[:10]:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
                story.append(Spacer(1, 5))
        else:
            story.append(Paragraph(str(recommendations), styles['Normal']))
        
        # Footer profesional
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX Professional Analysis System<br/>
        Sistema certificado para análisis biomecánico deportivo<br/>
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
# RUTAS PROFESIONALES
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "message": "InertiaX Professional API Running",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    })

@app.route("/upload", methods=["POST"])
def upload():
    """Endpoint profesional para carga de datos"""
    try:
        job_id = _ensure_job()
        session.modified = True

        # Datos del formulario profesional
        form = {
            "nombre_entrenador": request.form.get("nombre_entrenador", "").strip(),
            "origen_app": request.form.get("origen_app", "").strip(),
            "codigo_invitado": request.form.get("codigo_invitado", "").strip(),
        }

        log.info(f"📥 Solicitud de análisis profesional de: {form['nombre_entrenador']}")

        # Verificación de código premium
        code = form.get("codigo_invitado", "")
        payment_ok = False
        mensaje = None
        if code and code in app.config["GUEST_CODES"]:
            payment_ok = True
            mensaje = "🔓 ACCESO PREMIUM ACTIVADO - Análisis profesional disponible"

        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", error="❌ ARCHIVO NO ESPECIFICADO - Seleccione un archivo para análisis")

        if not _allowed_file(f.filename):
            return render_template("index.html", error="❌ FORMATO NO SOPORTADO - Use archivos CSV o Excel")

        # Procesamiento profesional del archivo
        ext = os.path.splitext(f.filename)[1].lower()
        safe_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(_job_dir(job_id), safe_name)
        f.save(save_path)

        # Metadatos profesionales
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "payment_ok": payment_ok,
            "form": form,
            "upload_time": datetime.now().isoformat()
        }
        _save_meta(job_id, meta)

        # Previsualización profesional
        try:
            df = parse_dataframe(save_path)
            df = preprocess_data_by_origin(df, form.get("origen_app", ""))
            
            # Generar tabla HTML profesional
            table_html = df.head(15).to_html(
                classes="table table-striped table-bordered table-hover table-sm",
                index=False,
                escape=False
            )
            
            log.info(f"✅ Previsualización generada: {len(df)} registros procesados")
            
            return render_template(
                "index.html",
                table_html=table_html,
                filename=f.filename,
                form_data=form,
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
    """Sistema de pago profesional"""
    if not mp:
        return jsonify(error="SISTEMA DE PAGO NO CONFIGURADO"), 500

    job_id = session.get("job_id")
    if not job_id:
        return jsonify(error="SESIÓN INVÁLIDA"), 400

    # Precio profesional por servicio premium
    price = 9900  # Servicio profesional premium

    pref_data = {
        "items": [{
            "title": "InertiaX Pro - Análisis Científico Premium",
            "description": "Análisis biomecánico profesional con IA científica",
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
        log.info("🚀 INICIANDO GENERACIÓN DE REPORTE PROFESIONAL")
        
        # 1. Carga y procesamiento profesional
        df = parse_dataframe(file_path)
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""))
        
        log.info(f"📊 Dataset profesional cargado: {df.shape[0]} registros")

        # 2. Análisis científico con IA
        log.info("🧠 EJECUTANDO ANÁLISIS CIENTÍFICO CON IA...")
        ai_result = run_professional_ai_analysis(df, meta.get("form", {}))
        
        # 3. Generación de gráficos profesionales
        log.info("📈 GENERANDO VISUALIZACIONES PROFESIONALES...")
        biomechanical_analysis = perform_advanced_biomechanical_analysis(df)
        professional_charts = generate_professional_charts(df, biomechanical_analysis)
        
        # 4. Gráficos adicionales de IA si están disponibles
        ai_charts = []
        python_code = ai_result.get("python_code_for_charts", "")
        if python_code:
            try:
                ai_charts = execute_ai_charts_code(python_code, df)
            except Exception as e:
                log.error(f"Error en gráficos IA: {e}")

        # 5. Generación de reporte PDF profesional
        log.info("📄 GENERANDO REPORTE PDF PROFESIONAL...")
        all_charts = professional_charts + ai_charts
        pdf_path = generate_professional_pdf(ai_result, all_charts, meta.get("form", {}))

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

        log.info("✅ REPORTE PROFESIONAL GENERADO EXITOSAMENTE")
            
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
        df = preprocess_data_by_origin(df, meta.get("form", {}).get("origen_app", ""))
        
        # Análisis científico
        biomechanical_analysis = perform_advanced_biomechanical_analysis(df)
        ai_result = run_professional_ai_analysis(df, meta.get("form", {}))
        
        return render_template(
            "preview.html",
            biomechanical_analysis=biomechanical_analysis,
            ai_analysis=ai_result.get("analysis", "Análisis científico en progreso..."),
            recommendations=ai_result.get("recommendations", []),
            filename=meta.get("file_name")
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
    log.info(f"🚀 INERTIAX PROFESSIONAL STARTING ON PORT {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
