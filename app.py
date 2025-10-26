from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
import numpy as np
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
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib import colors
from reportlab.lib.units import inch
import io

# ==============================
# CONFIGURACIÓN SIMPLE
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_simple_secure_key")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_pro")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# ==============================
# INICIALIZACIÓN
# ==============================

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app)

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inertiax_simple")

# Cliente OpenAI
ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

# ==============================
# FUNCIONES AUXILIARES
# ==============================

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

def parse_dataframe(path: str) -> pd.DataFrame:
    """Lee el archivo CSV o Excel"""
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)
    except Exception as e:
        log.error(f"Error leyendo archivo: {str(e)}")
        raise

# ==============================
# ANÁLISIS CON IA - SUPER SIMPLE
# ==============================

def analyze_with_ai(df: pd.DataFrame, nombre_entrenador: str, nombre_cliente: str) -> dict:
    """
    Analiza directamente el CSV con IA usando un prompt optimizado
    """
    if not ai_client:
        return get_fallback_analysis(df)
    
    try:
        # Preparar los datos para el análisis
        csv_content = df.to_csv(index=False)
        basic_stats = f"""
        Forma del dataset: {df.shape}
        Columnas: {list(df.columns)}
        Primeras filas:
        {df.head().to_string()}
        """
        
        # Prompt optimizado basado en tu ejemplo
        system_prompt = """
        Eres un especialista en análisis de datos deportivos (VBT - Velocity Based Training) 
        con experiencia en fuerza-velocidad y biomecánica. Analiza los datos y proporciona 
        un informe claro, accionable y profesional.

        **FORMATO DE RESPUESTA ESTRICTO EN JSON:**
        {
            "resumen_ejecutivo": "Breve resumen de 2-3 líneas",
            "analisis_detallado": "Análisis completo con secciones claras",
            "metricas_principales": {
                "velocidad_maxima": "valor y interpretación",
                "velocidad_promedio": "valor y interpretación", 
                "carga_maxima": "valor y interpretación",
                "fatiga_detectada": "valor y interpretación",
                "rm_estimado": "valor y rango"
            },
            "fortalezas": ["lista de fortalezas identificadas"],
            "areas_mejora": ["lista de áreas a mejorar"],
            "recomendaciones_entrenamiento": [
                {
                    "tipo": "Fuerza/Potencia/Etc",
                    "ejercicios": ["ej1", "ej2"],
                    "intensidad": "rango recomendado",
                    "objetivo": "qué se busca mejorar"
                }
            ],
            "plan_seguimiento": "Recomendaciones para próximas sesiones"
        }

        **ENFOQUE DEL ANÁLISIS:**
        1. Identifica patrones de fuerza-velocidad
        2. Analiza fatiga intra-sesión
        3. Detecta consistencia técnica
        4. Proporciona recomendaciones prácticas
        5. Estima 1RM cuando sea posible
        """

        user_prompt = f"""
        **DATOS PARA ANÁLISIS:**
        Entrenador: {nombre_entrenador}
        Cliente: {nombre_cliente}
        Fecha de análisis: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        **ESTADÍSTICAS BÁSICAS:**
        {basic_stats}
        
        **DATOS COMPLETOS (CSV):**
        ```csv
        {csv_content}
        ```
        
        **INSTRUCCIÓN:** 
        Analiza estos datos deportivos y proporciona un informe completo pero conciso, 
        enfocado en aplicaciones prácticas para el entrenamiento.
        """

        log.info("🧠 INICIANDO ANÁLISIS CON IA...")
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        log.info("✅ ANÁLISIS CON IA COMPLETADO")
        return result
        
    except Exception as e:
        log.error(f"❌ ERROR EN ANÁLISIS IA: {str(e)}")
        return get_fallback_analysis(df)

def get_fallback_analysis(df: pd.DataFrame) -> dict:
    """Análisis básico cuando no hay IA disponible"""
    return {
        "resumen_ejecutivo": "Análisis básico - Configure OPENAI_API_KEY para análisis avanzado",
        "analisis_detallado": f"Datos procesados: {df.shape[0]} registros, {df.shape[1]} columnas",
        "metricas_principales": {
            "registros_analizados": f"{df.shape[0]}",
            "columnas_detectadas": f"{list(df.columns)}"
        },
        "fortalezas": ["Datos cargados correctamente", "Formato válido detectado"],
        "areas_mejora": ["Active análisis IA para recomendaciones personalizadas"],
        "recomendaciones_entrenamiento": [
            {
                "tipo": "Configuración",
                "ejercicios": ["Configure API key OpenAI"],
                "intensidad": "N/A",
                "objetivo": "Habilitar análisis avanzado"
            }
        ],
        "plan_seguimiento": "Configure las variables de entorno necesarias"
    }

# ==============================
# GENERACIÓN DE GRÁFICOS SIMPLES
# ==============================

def generate_simple_charts(df: pd.DataFrame) -> List[BytesIO]:
    """Genera gráficos básicos y profesionales"""
    charts = []
    
    try:
        # Configuración de estilo
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Gráfico 1: Análisis de variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Gráfico de distribución de la primera variable numérica
            fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:3]):
                axes[i].hist(df[col].dropna(), bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribución de {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            charts.append(save_plot_to_buffer(fig))
        
        # Gráfico 2: Correlaciones si hay múltiples variables numéricas
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlaciones entre Variables', fontweight='bold')
            charts.append(save_plot_to_buffer(fig))
            
    except Exception as e:
        log.error(f"Error generando gráficos: {str(e)}")
    
    return charts

def save_plot_to_buffer(fig) -> BytesIO:
    """Guarda gráfico en buffer"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

# ==============================
# GENERACIÓN DE PDF PROFESIONAL
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera un PDF profesional y claro para el cliente"""
    
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_cliente_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos personalizados
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=30,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1A535C'),
            spaceAfter=12,
            fontName='Helvetica-Bold'
        )
        
        story = []
        
        # Header
        story.append(Paragraph("INFORME DE RENDIMIENTO DEPORTIVO", title_style))
        story.append(Spacer(1, 10))
        
        # Información básica
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'No especificado')}<br/>
        <b>Cliente:</b> {meta.get('nombre_cliente', 'No especificado')}<br/>
        <b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Archivo analizado:</b> {meta.get('file_name', 'No especificado')}<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Resumen Ejecutivo
        if 'resumen_ejecutivo' in ai_result:
            story.append(Paragraph("📊 RESUMEN EJECUTIVO", section_style))
            story.append(Paragraph(ai_result['resumen_ejecutivo'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Métricas Principales
        if 'metricas_principales' in ai_result:
            story.append(Paragraph("🎯 MÉTRICAS PRINCIPALES", section_style))
            metrics = ai_result['metricas_principales']
            for key, value in metrics.items():
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Gráficos
        if charts:
            story.append(Paragraph("📈 ANÁLISIS VISUAL", section_style))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts[:3]):  # Máximo 3 gráficos
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                except:
                    continue
        
        # Fortalezas y Áreas de Mejora
        col_width = doc.width / 2 - 10
        
        if 'fortalezas' in ai_result or 'areas_mejora' in ai_result:
            story.append(Paragraph("💪 EVALUACIÓN INTEGRAL", section_style))
            story.append(Spacer(1, 10))
            
            # Usar una tabla simple para dos columnas
            from reportlab.platypus import Table, TableStyle
            
            data = []
            
            # Fortalezas
            strengths_text = "<b>✅ FORTALEZAS:</b><br/>"
            if 'fortalezas' in ai_result and ai_result['fortalezas']:
                for strength in ai_result['fortalezas'][:5]:  # Máximo 5
                    strengths_text += f"• {strength}<br/>"
            else:
                strengths_text += "• Análisis en progreso<br/>"
            
            # Áreas de mejora
            improvements_text = "<b>🎯 ÁREAS DE MEJORA:</b><br/>"
            if 'areas_mejora' in ai_result and ai_result['areas_mejora']:
                for improvement in ai_result['areas_mejora'][:5]:  # Máximo 5
                    improvements_text += f"• {improvement}<br/>"
            else:
                improvements_text += "• Análisis en progreso<br/>"
            
            data = [[Paragraph(strengths_text, styles['Normal']), 
                    Paragraph(improvements_text, styles['Normal'])]]
            
            table = Table(data, colWidths=[col_width, col_width])
            table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
                ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 15))
        
        # Recomendaciones de Entrenamiento
        if 'recomendaciones_entrenamiento' in ai_result:
            story.append(Paragraph("🏋️‍♂️ PLAN DE ENTRENAMIENTO", section_style))
            recommendations = ai_result['recomendaciones_entrenamiento']
            
            if isinstance(recommendations, list):
                for rec in recommendations[:4]:  # Máximo 4 recomendaciones
                    if isinstance(rec, dict):
                        story.append(Paragraph(f"<b>{rec.get('tipo', 'General')}:</b>", styles['Normal']))
                        if 'ejercicios' in rec:
                            exercises = ', '.join(rec['ejercicios'][:3])  # Máximo 3 ejercicios
                            story.append(Paragraph(f"Ejercicios: {exercises}", styles['Normal']))
                        if 'intensidad' in rec:
                            story.append(Paragraph(f"Intensidad: {rec['intensidad']}", styles['Normal']))
                        if 'objetivo' in rec:
                            story.append(Paragraph(f"Objetivo: {rec['objetivo']}", styles['Normal']))
                        story.append(Spacer(1, 8))
            story.append(Spacer(1, 10))
        
        # Plan de Seguimiento
        if 'plan_seguimiento' in ai_result:
            story.append(Paragraph("📅 PRÓXIMOS PASOS", section_style))
            story.append(Paragraph(ai_result['plan_seguimiento'], styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX - Sistema de Análisis Deportivo<br/>
        Tecnología aplicada al alto rendimiento</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF: {str(e)}")
        # PDF de error simple
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX - ERROR EN REPORTE")
        c.drawString(100, 730, "Contacte al soporte técnico")
        c.save()
        return error_path

# ==============================
# RUTAS PRINCIPALES
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Endpoint simple para subir archivo y generar reporte"""
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({"error": "Archivo no especificado"}), 400
    
    if not _allowed_file(f.filename):
        return jsonify({"error": "Formato no soportado"}), 400
    
    try:
        # Crear job único
        job_id = _ensure_job()
        session.modified = True
        
        # Guardar archivo
        ext = os.path.splitext(f.filename)[1].lower()
        safe_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(_job_dir(job_id), safe_name)
        f.save(save_path)
        
        # Obtener datos del formulario
        nombre_entrenador = request.form.get('nombre_entrenador', 'Entrenador')
        nombre_cliente = request.form.get('nombre_cliente', 'Cliente')
        
        # Procesar archivo
        df = parse_dataframe(save_path)
        
        # Guardar metadata
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "nombre_entrenador": nombre_entrenador,
            "nombre_cliente": nombre_cliente,
            "upload_time": datetime.now().isoformat(),
            "data_shape": f"{df.shape}",
            "columns": list(df.columns)
        }
        _save_meta(job_id, meta)
        
        # Análisis con IA
        ai_result = analyze_with_ai(df, nombre_entrenador, nombre_cliente)
        
        # Generar gráficos
        charts = generate_simple_charts(df)
        
        # Generar PDF
        pdf_path = generate_professional_pdf(ai_result, charts, meta)
        
        # Crear ZIP con resultados
        zip_path = os.path.join(_job_dir(job_id), f"reporte_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "Reporte_Rendimiento.pdf")
            zf.write(save_path, f"datos_originales/{f.filename}")
            
            # Agregar datos procesados
            processed_path = os.path.join(_job_dir(job_id), "datos_procesados.csv")
            df.to_csv(processed_path, index=False)
            zf.write(processed_path, "datos_procesados/analisis.csv")
            
            # Agregar resumen en JSON
            summary_path = os.path.join(_job_dir(job_id), "resumen_analisis.json")
            with open(summary_path, "w") as sf:
                json.dump(ai_result, sf, indent=2, ensure_ascii=False)
            zf.write(summary_path, "resumen_analisis/resumen.json")

        # Limpieza
        try:
            os.remove(pdf_path)
            os.remove(processed_path)
            os.remove(summary_path)
        except:
            pass

        log.info("✅ REPORTE GENERADO EXITOSAMENTE")
        
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"Reporte_InertiaX_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ ERROR: {str(e)}")
        return jsonify({"error": f"Error procesando archivo: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "InertiaX Simple API Running"})

# Manejo de errores
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Archivo demasiado grande"}), 413

@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX SIMPLE STARTING ON PORT {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
