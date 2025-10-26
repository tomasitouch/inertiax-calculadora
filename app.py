from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from io import BytesIO
from typing import Dict, List
from datetime import datetime
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from flask import Flask, render_template, request, send_file, session, jsonify
from flask_cors import CORS
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# ==============================
# CONFIGURACIÓN MEJORADA
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_pro_key")
    UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/inertiax_pro")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    # Timeout más conservador para Render
    REQUEST_TIMEOUT = 45

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inertiax_pro")

ai_client = OpenAI(api_key=app.config["OPENAI_API_KEY"], timeout=app.config["REQUEST_TIMEOUT"]) if app.config["OPENAI_API_KEY"] else None

# Executor para procesamiento en segundo plano
executor = ThreadPoolExecutor(max_workers=2)

# ==============================
# FUNCIONES AUXILIARES
# ==============================

def _job_dir(job_id: str) -> str:
    d = os.path.join(app.config["UPLOAD_DIR"], job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _save_meta(job_id: str, meta: Dict) -> str:
    p = os.path.join(_job_dir(job_id), "meta.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return p

def _load_meta(job_id: str) -> Dict:
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
# ANÁLISIS CON IA - PROMPT MEJORADO
# ==============================

def analyze_with_ai_and_charts(df: pd.DataFrame, nombre_entrenador: str, nombre_cliente: str) -> dict:
    """
    Analiza los datos con un prompt optimizado para análisis detallado
    """
    if not ai_client:
        return get_fallback_analysis(df)
    
    try:
        csv_content = df.to_csv(index=False)
        basic_info = f"""
        Dataset: {df.shape[0]} filas, {df.shape[1]} columnas
        Columnas: {list(df.columns)}
        Tipos de datos: {dict(df.dtypes)}
        Primeras filas:
        {df.head().to_string()}
        """
        
        # PROMPT MEJORADO - ANÁLISIS DETALLADO SIN RESUMIR
        system_prompt = """
        Eres un CONSULTOR DEPORTIVO ELITE especializado en VBT (Velocity Based Training), 
        biomecánica y análisis de rendimiento deportivo. 

        **INSTRUCCIONES CRÍTICAS:**
        - NO RESUMIR información
        - PROPORCIONAR ANÁLISIS EXTENSO Y DETALLADO
        - EXPLICAR patrones, tendencias y anomalías
        - ANALIZAR repetición por repetición cuando sea posible
        - PROPORCIONAR INTERPRETACIÓN BIOMECÁNICA COMPLETA
        - INCLUIR RECOMENDACIONES ESPECÍFICAS Y ACCIONABLES

        **ESTILO DE RESPUESTA:**
        - Texto extenso y narrativo
        - Explicaciones pedagógicas
        - Análisis técnico profundo
        - Como si estuvieras consultando con un entrenador experto

        **FORMATO DE RESPUESTA JSON:**
        {
            "analisis_completo": "Texto EXTENSO con análisis completo de todos los aspectos...",
            "metricas_detalladas": {
                "velocidad_maxima": "Valor y explicación extensa...",
                "velocidad_promedio": "Valor y explicación extensa...",
                "carga_maxima": "Valor y explicación extensa...",
                "fatiga_intra_serie": "Cálculo y explicación extensa...",
                "consistencia_tecnica": "Evaluación extensa...",
                "rm_estimado": "Cálculo y rango con explicación..."
            },
            "patrones_detectados": [
                "Patrón 1 con explicación extensa...",
                "Patrón 2 con explicación extensa...",
                "Patrón 3 con explicación extensa..."
            ],
            "analisis_por_repeticion": "Análisis EXTENSO repetición por repetición...",
            "curva_fuerza_velocidad": "Análisis EXTENSO de la relación fuerza-velocidad...",
            "evaluacion_rendimiento": "Evaluación EXTENSA del rendimiento general...",
            "recomendaciones_especificas": [
                {
                    "categoria": "Fuerza/Potencia/Etc",
                    "accion": "Recomendación específica y detallada",
                    "ejercicios": ["ej1", "ej2", "ej3"],
                    "series_reps": "Prescripción específica",
                    "intensidad": "Rango detallado",
                    "objetivo": "Objetivo fisiológico explicado",
                    "fundamento": "Explicación técnica extensa del porqué"
                }
            ],
            "plan_entrenamiento": "Plan de entrenamiento EXTENSO y personalizado...",
            "mensajes_coaching": [
                "Mensaje 1 para el atleta",
                "Mensaje 2 para el atleta", 
                "Mensaje 3 para el atleta"
            ],
            "codigo_graficos": "Código Python completo para gráficos profesionales",
            "descripcion_graficos": "Descripción detallada de cada gráfico generado"
        }

        **ENFOQUE EN ANÁLISIS DEPORTIVO:**
        1. Relación carga-velocidad y perfil fuerza-velocidad
        2. Fatiga intra-serie e inter-serie
        3. Consistencia técnica y control motor
        4. Potencia y producción de fuerza
        5. Estimación de 1RM y capacidades máximas
        6. Detección de patrones de rendimiento
        7. Recomendaciones específicas por cualidad física
        """

        user_prompt = f"""
        **CONTEXTO PROFESIONAL:**
        - Entrenador: {nombre_entrenador}
        - Cliente/Atleta: {nombre_cliente}  
        - Fecha de análisis: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        - Sistema: InertiaX Pro - Análisis Deportivo con IA

        **INFORMACIÓN TÉCNICA DEL DATASET:**
        {basic_info}

        **DATOS COMPLETOS PARA ANÁLISIS:**
        ```csv
        {csv_content}
        ```

        **INSTRUCCIONES FINALES:**
        Realiza un análisis COMPLETO Y EXTENSO como consultor deportivo profesional.
        NO OMITAS DETALLES. NO RESUMAS.
        Proporciona el máximo valor consultivo posible.

        Incluye:
        - Análisis repetición por repetición
        - Cálculos de fatiga y consistencia  
        - Interpretación biomecánica
        - Recomendaciones específicas por cualidad física
        - Plan de entrenamiento personalizado
        - Mensajes de coaching para el atleta
        - Código para gráficos profesionales

        **IMPORTANTE:** Tu respuesta debe ser EXTENSA y DETALLADA, similar a una consultoría profesional presencial.
        """

        log.info("🧠 INICIANDO ANÁLISIS DETALLADO CON IA...")
        
        # Timeout más conservador
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=12000,  # Suficiente para análisis extenso
            response_format={"type": "json_object"},
            timeout=30  # Timeout específico para OpenAI
        )
        
        result = json.loads(response.choices[0].message.content)
        log.info("✅ ANÁLISIS DETALLADO COMPLETADO")
        return result
        
    except Exception as e:
        log.error(f"❌ ERROR EN ANÁLISIS IA: {str(e)}")
        return get_fallback_analysis(df)

def get_fallback_analysis(df: pd.DataFrame) -> dict:
    return {
        "analisis_completo": f"Análisis básico - Configure OPENAI_API_KEY para análisis avanzado con IA\n\nDatos procesados: {df.shape[0]} registros, {df.shape[1]} variables\nColumnas: {list(df.columns)}",
        "metricas_detalladas": {
            "registros_analizados": f"{df.shape[0]}",
            "variables_detectadas": f"{list(df.columns)}",
            "configuracion_requerida": "Configure OPENAI_API_KEY para análisis completo"
        },
        "patrones_detectados": [
            "Active análisis IA para detección de patrones",
            "Configure variables de entorno necesarias"
        ],
        "analisis_por_repeticion": "Análisis por repetición disponible con IA activada",
        "curva_fuerza_velocidad": "Análisis F-V disponible con IA activada",
        "evaluacion_rendimiento": "Evaluación completa disponible con IA activada",
        "recomendaciones_especificas": [
            {
                "categoria": "Configuración",
                "accion": "Configurar API key de OpenAI",
                "ejercicios": ["Configuración del sistema"],
                "series_reps": "N/A",
                "intensidad": "N/A",
                "objetivo": "Habilitar análisis avanzado",
                "fundamento": "Necesario para análisis deportivo con IA"
            }
        ],
        "plan_entrenamiento": "Configure OPENAI_API_KEY para plan personalizado",
        "mensajes_coaching": [
            "Active el análisis con IA para recomendaciones específicas",
            "Configure las credenciales necesarias"
        ],
        "codigo_graficos": "# Gráficos disponibles con IA activada\n# Configure OPENAI_API_KEY",
        "descripcion_graficos": "Gráficos profesionales generados automáticamente con IA"
    }

# ==============================
# GENERACIÓN DE GRÁFICOS (igual que antes)
# ==============================

def generate_charts_from_code(python_code: str, df: pd.DataFrame) -> List[BytesIO]:
    charts = []
    
    if not python_code or "Configure OPENAI_API_KEY" in python_code:
        return charts
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        local_vars = {
            'df': df,
            'plt': plt,
            'sns': sns,
            'np': np,
            'BytesIO': BytesIO,
            'charts': []
        }
        
        exec(python_code, local_vars)
        
        charts = local_vars.get('charts', [])
        valid_charts = []
        for chart in charts:
            if isinstance(chart, BytesIO):
                valid_charts.append(chart)
        
        log.info(f"📊 {len(valid_charts)} gráficos generados por IA")
        return valid_charts[:4]
        
    except Exception as e:
        log.error(f"❌ ERROR EJECUTANDO CÓDIGO DE GRÁFICOS: {str(e)}")
        return generate_fallback_charts(df)

def generate_fallback_charts(df: pd.DataFrame) -> List[BytesIO]:
    charts = []
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        
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
# GENERACIÓN DE PDF MEJORADO
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera PDF profesional con análisis extenso"""
    
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
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
        story.append(Paragraph("INFORME DE RENDIMIENTO DEPORTIVO - INERTIAX PRO", title_style))
        story.append(Spacer(1, 10))
        
        # Información básica
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'No especificado')}<br/>
        <b>Cliente/Atleta:</b> {meta.get('nombre_cliente', 'No especificado')}<br/>
        <b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Archivo analizado:</b> {meta.get('file_name', 'No especificado')}<br/>
        <b>Sistema:</b> InertiaX Pro - Análisis Deportivo con IA<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Análisis Completo
        if 'analisis_completo' in ai_result:
            story.append(Paragraph("🔍 ANÁLISIS COMPLETO DEL RENDIMIENTO", section_style))
            analysis_text = ai_result['analisis_completo'].replace('\n', '<br/>')
            story.append(Paragraph(analysis_text, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Métricas Detalladas
        if 'metricas_detalladas' in ai_result:
            story.append(Paragraph("📊 MÉTRICAS DETALLADAS", section_style))
            metrics = ai_result['metricas_detalladas']
            for key, value in metrics.items():
                story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Gráficos
        if charts:
            story.append(Paragraph("📈 ANÁLISIS VISUAL", section_style))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts):
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 10))
                    
                    desc = f"Figura {i+1}: {ai_result.get('descripcion_graficos', 'Análisis visual profesional')}"
                    story.append(Paragraph(desc, styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Patrones Detectados
        if 'patrones_detectados' in ai_result and ai_result['patrones_detectados']:
            story.append(Paragraph("🎯 PATRONES DETECTADOS", section_style))
            for pattern in ai_result['patrones_detectados'][:6]:
                story.append(Paragraph(f"• {pattern}", styles['Normal']))
                story.append(Spacer(1, 5))
            story.append(Spacer(1, 15))
        
        # Recomendaciones Específicas
        if 'recomendaciones_especificas' in ai_result:
            story.append(Paragraph("🏋️‍♂️ RECOMENDACIONES ESPECÍFICAS", section_style))
            for rec in ai_result['recomendaciones_especificas'][:4]:
                if isinstance(rec, dict):
                    story.append(Paragraph(f"<b>{rec.get('categoria', 'General')}:</b>", styles['Normal']))
                    story.append(Paragraph(f"<b>Acción:</b> {rec.get('accion', '')}", styles['Normal']))
                    if 'ejercicios' in rec:
                        exercises = ', '.join(rec['ejercicios'])
                        story.append(Paragraph(f"<b>Ejercicios:</b> {exercises}", styles['Normal']))
                    if 'series_reps' in rec:
                        story.append(Paragraph(f"<b>Prescripción:</b> {rec['series_reps']}", styles['Normal']))
                    if 'fundamento' in rec:
                        story.append(Paragraph(f"<b>Fundamento:</b> {rec['fundamento']}", styles['Normal']))
                    story.append(Spacer(1, 10))
            story.append(Spacer(1, 15))
        
        # Mensajes de Coaching
        if 'mensajes_coaching' in ai_result and ai_result['mensajes_coaching']:
            story.append(Paragraph("💬 MENSAJES DE COACHING PARA EL ATLETA", section_style))
            for msg in ai_result['mensajes_coaching'][:5]:
                story.append(Paragraph(f"• {msg}", styles['Normal']))
                story.append(Spacer(1, 5))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = """
        <i>Reporte generado por InertiaX Pro - Sistema de Análisis Deportivo con IA<br/>
        Tecnología aplicada al alto rendimiento - www.inertiax.pro</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF: {str(e)}")
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "INERTIAX PRO - ERROR EN REPORTE")
        c.drawString(100, 730, "Contacte al soporte técnico")
        c.save()
        return error_path

# ==============================
# RUTAS PRINCIPALES - MEJORADAS
# ==============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Endpoint principal con manejo mejorado de timeouts"""
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({"error": "Archivo no especificado"}), 400
    
    try:
        # Configurar job rápidamente
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
        
        # Guardar metadata inicial
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "nombre_entrenador": nombre_entrenador,
            "nombre_cliente": nombre_cliente,
            "upload_time": datetime.now().isoformat(),
            "status": "processing"
        }
        _save_meta(job_id, meta)
        
        log.info("🚀 INICIANDO ANÁLISIS COMPLETO...")
        
        # Análisis con IA (punto crítico de timeout)
        ai_result = analyze_with_ai_and_charts(df, nombre_entrenador, nombre_cliente)
        
        # Generar gráficos
        log.info("📊 GENERANDO GRÁFICOS...")
        charts = generate_charts_from_code(ai_result.get('codigo_graficos', ''), df)
        
        # Generar PDF
        log.info("📄 GENERANDO PDF PROFESIONAL...")
        pdf_path = generate_professional_pdf(ai_result, charts, meta)
        
        # Crear ZIP con resultados
        zip_path = os.path.join(_job_dir(job_id), f"reporte_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "Reporte_Rendimiento_InertiaX_Pro.pdf")
            zf.write(save_path, f"datos_originales/{f.filename}")
            
            # Agregar análisis completo en JSON
            analysis_path = os.path.join(_job_dir(job_id), "analisis_completo.json")
            with open(analysis_path, "w", encoding="utf-8") as af:
                json.dump(ai_result, af, indent=2, ensure_ascii=False)
            zf.write(analysis_path, "analisis/resultado_completo.json")

        # Limpieza
        try:
            os.remove(pdf_path)
            os.remove(analysis_path)
        except:
            pass

        log.info("✅ REPORTE PROFESIONAL GENERADO EXITOSAMENTE!")
        
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"Reporte_InertiaX_Pro_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ ERROR EN PROCESAMIENTO: {str(e)}")
        return jsonify({"error": f"Error procesando archivo: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "message": "InertiaX Pro API Running",
        "timestamp": datetime.now().isoformat()
    })

# Manejo de errores mejorado
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Archivo demasiado grande - Máximo 50MB"}), 413

@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Error interno del servidor"}), 500

@app.errorhandler(408)
def timeout_error(_e):
    return jsonify({"error": "Timeout en el procesamiento"}), 408

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX PRO STARTING ON PORT {port}")
    # Deshabilitar debug para producción
    app.run(host="0.0.0.0", port=port, debug=False)
