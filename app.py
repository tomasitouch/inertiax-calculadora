from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from io import BytesIO
from typing import Dict, List
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, send_file, session
from flask_cors import CORS
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib import colors
from reportlab.lib.units import inch
import base64
import requests

# ==============================
# CONFIGURACIÓN
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_simple_key")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_pro")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # Usar gpt-4o que es muy bueno para análisis y puede generar código de gráficos
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inertiax_simple")

ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"]) if app.config["OPENAI_API_KEY"] else None

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _save_meta(job_id: str, meta: Dict) -> str:
    import json
    p = os.path.join(_job_dir(job_id), "meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p

def _load_meta(job_id: str) -> Dict:
    import json
    p = os.path.join(_job_dir(job_id), "meta.json")
    if not os.path.exists(p):
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_job() -> str:
    if "job_id" not in session:
        session["job_id"] = uuid.uuid4().hex
    return session["job_id"]

def parse_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)

# ==============================
# ANÁLISIS CON IA + GRÁFICOS
# ==============================

def analyze_with_ai_and_charts(df: pd.DataFrame, nombre_entrenador: str, nombre_cliente: str) -> dict:
    """
    Analiza los datos y genera gráficos usando IA
    """
    if not ai_client:
        return get_fallback_analysis(df)
    
    try:
        csv_content = df.to_csv(index=False)
        basic_info = f"""
        Dataset: {df.shape[0]} filas, {df.shape[1]} columnas
        Columnas: {list(df.columns)}
        Tipos de datos: {dict(df.dtypes)}
        """
        
        # Prompt optimizado para análisis deportivo + generación de gráficos
        system_prompt = """
        Eres un especialista en análisis de datos deportivos (VBT - Velocity Based Training) 
        y visualización de datos. Tu tarea es:

        1. ANALIZAR los datos deportivos proporcionados
        2. GENERAR código Python para crear gráficos profesionales
        3. PROPORCIONAR un análisis completo en formato JSON

        **INSTRUCCIONES PARA GRÁFICOS:**
        - Genera código Python usando matplotlib/seaborn
        - Los gráficos deben ser profesionales y claros
        - Máximo 4 gráficos
        - Incluir títulos, labels, y styling profesional
        - Guardar cada gráfico en un BytesIO

        **FORMATO DE RESPUESTA JSON:**
        {
            "resumen_ejecutivo": "Resumen de 2-3 líneas",
            "analisis_detallado": "Análisis completo por secciones",
            "metricas_clave": {
                "metric1": "valor y explicación",
                "metric2": "valor y explicación"
            },
            "fortalezas": ["lista de fortalezas"],
            "areas_mejora": ["lista de áreas a mejorar"],
            "recomendaciones": [
                {
                    "categoria": "Fuerza/Potencia/etc",
                    "accion": "Recomendación específica",
                    "detalles": "Explicación técnica"
                }
            ],
            "codigo_graficos": "código Python completo para generar gráficos",
            "descripcion_graficos": "Descripción de qué muestra cada gráfico"
        }
        """

        user_prompt = f"""
        **CONTEXTO:**
        Entrenador: {nombre_entrenador}
        Cliente: {nombre_cliente}
        Fecha: {datetime.now().strftime('%Y-%m-%d')}

        **INFORMACIÓN DEL DATASET:**
        {basic_info}

        **DATOS COMPLETOS (CSV):**
        ```csv
        {csv_content}
        ```

        **INSTRUCCIONES ESPECÍFICAS:**
        1. Analiza patrones de rendimiento deportivo
        2. Identifica métricas clave (velocidad, carga, fatiga, etc.)
        3. Genera código para 3-4 gráficos profesionales
        4. Proporciona recomendaciones prácticas

        Los gráficos deben incluir:
        - Análisis de tendencias principales
        - Distribución de métricas clave  
        - Relaciones entre variables importantes
        - Evolución temporal si hay datos de fecha

        **IMPORTANTE:** El código de gráficos debe usar:
        - matplotlib y seaborn
        - Guardar en BytesIO objects
        - Ser ejecutable directamente
        """

        log.info("🧠 INICIANDO ANÁLISIS COMPLETO CON IA...")
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=8000,  # Más tokens para incluir código
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        log.info("✅ ANÁLISIS Y CÓDIGO DE GRÁFICOS GENERADO")
        return result
        
    except Exception as e:
        log.error(f"❌ ERROR EN ANÁLISIS IA: {str(e)}")
        return get_fallback_analysis(df)

def get_fallback_analysis(df: pd.DataFrame) -> dict:
    return {
        "resumen_ejecutivo": "Configure OPENAI_API_KEY para análisis avanzado con IA",
        "analisis_detallado": f"Datos básicos: {df.shape[0]} registros, {df.shape[1]} variables",
        "metricas_clave": {
            "registros": f"{df.shape[0]}",
            "variables": f"{list(df.columns)}"
        },
        "fortalezas": ["Datos cargados correctamente"],
        "areas_mejora": ["Active IA para análisis completo"],
        "recomendaciones": [
            {
                "categoria": "Configuración",
                "accion": "Configurar API key de OpenAI",
                "detalles": "Necesario para análisis avanzado"
            }
        ],
        "codigo_graficos": "# Gráficos no disponibles sin IA\n# Configure OPENAI_API_KEY",
        "descripcion_graficos": "Gráficos generados automáticamente por IA"
    }

def generate_charts_from_code(python_code: str, df: pd.DataFrame) -> List[BytesIO]:
    """
    Ejecuta el código Python generado por IA para crear gráficos
    """
    charts = []
    
    if not python_code or "Configure OPENAI_API_KEY" in python_code:
        return charts
    
    try:
        # Preparar el entorno de ejecución
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        
        # Configuración básica
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Variables disponibles para el código
        local_vars = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'np': np,
            'BytesIO': BytesIO,
            'charts': []
        }
        
        # Ejecutar el código generado por IA
        exec(python_code, local_vars)
        
        # Obtener los gráficos generados
        charts = local_vars.get('charts', [])
        
        # Asegurarse de que sean BytesIO
        valid_charts = []
        for chart in charts:
            if isinstance(chart, BytesIO):
                valid_charts.append(chart)
        
        log.info(f"📊 {len(valid_charts)} gráficos generados por IA")
        return valid_charts[:4]  # Máximo 4 gráficos
        
    except Exception as e:
        log.error(f"❌ ERROR EJECUTANDO CÓDIGO DE GRÁFICOS: {str(e)}")
        # Generar gráficos básicos de respaldo
        return generate_fallback_charts(df)

def generate_fallback_charts(df: pd.DataFrame) -> List[BytesIO]:
    """Gráficos básicos si falla el código de IA"""
    charts = []
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from io import BytesIO
        
        # Gráfico 1: Distribución de variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(2, len(numeric_cols)), figsize=(12, 5))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:2]):
                axes[i].hist(df[col].dropna(), bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribución de {col}', fontweight='bold')
                axes[i].set_xlabel(col)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts.append(buf)
            plt.close()
        
        # Gráfico 2: Correlaciones si hay múltiples variables
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlaciones entre Variables', fontweight='bold')
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            charts.append(buf)
            plt.close()
            
    except Exception as e:
        log.error(f"Error en gráficos de respaldo: {e}")
    
    return charts

# ==============================
# GENERACIÓN DE PDF
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera PDF profesional con análisis y gráficos"""
    
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        # Estilos
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        
        section_style = ParagraphStyle(
            'SectionStyle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#1A535C'),
            spaceAfter=10,
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
        <b>Fecha:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Archivo:</b> {meta.get('file_name', 'No especificado')}<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Resumen Ejecutivo
        if 'resumen_ejecutivo' in ai_result:
            story.append(Paragraph("📊 RESUMEN EJECUTIVO", section_style))
            story.append(Paragraph(ai_result['resumen_ejecutivo'], styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Métricas Clave
        if 'metricas_clave' in ai_result:
            story.append(Paragraph("🎯 MÉTRICAS PRINCIPALES", section_style))
            metrics = ai_result['metricas_clave']
            for key, value in metrics.items():
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Gráficos generados por IA
        if charts:
            story.append(Paragraph("📈 ANÁLISIS VISUAL", section_style))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts):
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    
                    # Descripción del gráfico
                    desc = f"Figura {i+1}: {ai_result.get('descripcion_graficos', 'Análisis visual generado por IA')}"
                    story.append(Paragraph(desc, styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Fortalezas y Áreas de Mejora
        if 'fortalezas' in ai_result or 'areas_mejora' in ai_result:
            story.append(Paragraph("💪 EVALUACIÓN INTEGRAL", section_style))
            
            col1 = "<b>✅ FORTALEZAS:</b><br/>"
            if 'fortalezas' in ai_result:
                for strength in ai_result['fortalezas'][:4]:
                    col1 += f"• {strength}<br/>"
            
            col2 = "<b>🎯 ÁREAS DE MEJORA:</b><br/>"
            if 'areas_mejora' in ai_result:
                for area in ai_result['areas_mejora'][:4]:
                    col2 += f"• {area}<br/>"
            
            from reportlab.platypus import Table, TableStyle
            data = [[Paragraph(col1, styles['Normal']), Paragraph(col2, styles['Normal'])]]
            table = Table(data, colWidths=[doc.width/2-10, doc.width/2-10])
            table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            
            story.append(table)
            story.append(Spacer(1, 15))
        
        # Recomendaciones
        if 'recomendaciones' in ai_result:
            story.append(Paragraph("🏋️‍♂️ RECOMENDACIONES", section_style))
            for rec in ai_result['recomendaciones'][:3]:
                if isinstance(rec, dict):
                    story.append(Paragraph(f"<b>{rec.get('categoria', 'General')}:</b> {rec.get('accion', '')}", styles['Normal']))
                    if 'detalles' in rec:
                        story.append(Paragraph(f"<i>{rec['detalles']}</i>", styles['Italic']))
                    story.append(Spacer(1, 8))
            story.append(Spacer(1, 10))
        
        # Análisis Detallado
        if 'analisis_detallado' in ai_result:
            story.append(Paragraph("🔍 ANÁLISIS DETALLADO", section_style))
            analysis = ai_result['analisis_detallado'].replace('\n', '<br/>')
            story.append(Paragraph(analysis, styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = "<i>Reporte generado por InertiaX - Análisis con IA</i>"
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF: {str(e)}")
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX - ERROR EN REPORTE")
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
    """Endpoint principal - subir archivo y generar reporte"""
    if 'file' not in request.files:
        return {"error": "No se envió archivo"}, 400
    
    f = request.files['file']
    if not f or f.filename == '':
        return {"error": "Archivo no especificado"}, 400
    
    try:
        # Configurar job
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
            "upload_time": datetime.now().isoformat()
        }
        _save_meta(job_id, meta)
        
        # Análisis completo con IA (incluye código de gráficos)
        log.info("🚀 INICIANDO ANÁLISIS COMPLETO CON IA...")
        ai_result = analyze_with_ai_and_charts(df, nombre_entrenador, nombre_cliente)
        
        # Generar gráficos desde el código de IA
        log.info("📊 GENERANDO GRÁFICOS DESDE CÓDIGO IA...")
        charts = generate_charts_from_code(ai_result.get('codigo_graficos', ''), df)
        
        # Generar PDF profesional
        log.info("📄 GENERANDO PDF PROFESIONAL...")
        pdf_path = generate_professional_pdf(ai_result, charts, meta)
        
        # Crear ZIP con resultados
        zip_path = os.path.join(_job_dir(job_id), f"reporte_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "Reporte_Rendimiento.pdf")
            zf.write(save_path, f"datos_originales/{f.filename}")
            
            # Agregar análisis en JSON
            analysis_path = os.path.join(_job_dir(job_id), "analisis_ia.json")
            with open(analysis_path, "w") as af:
                json.dump(ai_result, af, indent=2, ensure_ascii=False)
            zf.write(analysis_path, "analisis/resultado_ia.json")

        # Limpieza
        try:
            os.remove(pdf_path)
            os.remove(analysis_path)
        except:
            pass

        log.info("✅ REPORTE COMPLETO GENERADO!")
        
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"Reporte_InertiaX_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ ERROR: {str(e)}")
        return {"error": f"Error procesando archivo: {str(e)}"}, 500

@app.route("/health")
def health():
    return {"status": "ok", "message": "InertiaX Simple API Running"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX SIMPLE CON IA STARTING ON PORT {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
