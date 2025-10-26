from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from io import BytesIO
from typing import Dict, List
from datetime import datetime
import time

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
# CONFIGURACIÓN OPTIMIZADA
# ==============================

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "inertiax_pro_key")
    UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ALLOWED_EXT = {".csv", ".xls", ".xlsx"}
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Modelo balanceado

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.from_object(Config)
app.secret_key = app.config["SECRET_KEY"]
CORS(app)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inertiax_pro")

# Cliente OpenAI optimizado
ai_client = None
if app.config["OPENAI_API_KEY"]:
    try:
        ai_client = OpenAI(
            api_key=app.config["OPENAI_API_KEY"],
            timeout=30.0,  # Timeout balanceado
            max_retries=1
        )
        log.info("✅ Cliente OpenAI configurado")
    except Exception as e:
        log.error(f"❌ Error configurando OpenAI: {e}")
        ai_client = None

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
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        else:
            return pd.read_excel(path)
    except Exception as e:
        log.error(f"Error leyendo archivo {path}: {e}")
        raise

# ==============================
# ANÁLISIS CON IA - OPTIMIZADO
# ==============================

def analyze_with_ai_complete(df: pd.DataFrame, nombre_entrenador: str, nombre_cliente: str) -> dict:
    """
    Análisis completo usando TODOS los datos y generando gráficos personalizados
    """
    if not ai_client:
        return get_fallback_analysis(df)
    
    try:
        # USAR TODOS LOS DATOS - sin muestreo
        csv_content = df.to_csv(index=False)
        
        # Información completa del dataset
        basic_info = f"""
        Dataset completo: {df.shape[0]} filas, {df.shape[1]} columnas
        Columnas: {list(df.columns)}
        Tipos de datos: {dict(df.dtypes)}
        Estadísticas básicas:
        {df.describe().to_string()}
        """
        
        # Prompt OPTIMIZADO para análisis deportivo + gráficos personalizados
        system_prompt = """Eres un especialista en análisis de datos deportivos y visualización.
        
        Tu tarea es:
        1. Analizar los datos deportivos COMPLETOS
        2. Generar código Python para gráficos PERSONALIZADOS y RELEVANTES
        3. Proporcionar un análisis claro y accionable
        
        **INSTRUCCIONES PARA GRÁFICOS:**
        - Usa TODOS los datos del dataset (no muestrees)
        - Genera gráficos específicos para el tipo de datos deportivos
        - Máximo 3-4 gráficos más relevantes
        - Incluye: matplotlib, seaborn, estilo profesional
        - Cada gráfico debe guardarse en BytesIO y agregarse a la lista 'charts'
        
        **FORMATO DE RESPUESTA JSON:**
        {
            "analisis_completo": "Análisis general del rendimiento...",
            "hallazgos_principales": ["Hallazgo 1", "Hallazgo 2", "Hallazgo 3"],
            "metricas_clave": {
                "metric1": "valor + explicación",
                "metric2": "valor + explicación"
            },
            "recomendaciones": [
                "Recomendación práctica 1",
                "Recomendación práctica 2", 
                "Recomendación práctica 3"
            ],
            "codigo_graficos": "código Python completo para gráficos personalizados",
            "descripcion_graficos": "Descripción de qué muestra cada gráfico"
        }"""

        user_prompt = f"""
        **CONTEXTO:**
        - Entrenador: {nombre_entrenador}
        - Atleta: {nombre_cliente}  
        - Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        - Dataset COMPLETO: {df.shape[0]} registros

        **INFORMACIÓN DEL DATASET:**
        {basic_info}

        **DATOS COMPLETOS (CSV):**
        ```csv
        {csv_content}
        ```

        **INSTRUCCIONES ESPECÍFICAS:**
        1. Analiza TODO el dataset (no omitas datos)
        2. Identifica patrones deportivos relevantes
        3. Genera código para gráficos PERSONALIZADOS (no genéricos)
        4. Los gráficos deben usar todos los datos y ser específicos para este dataset
        5. Proporciona recomendaciones prácticas basadas en el análisis completo

        **EJEMPLOS DE GRÁFICOS DEPORTIVOS:**
        - Curva fuerza-velocidad
        - Evolución de rendimiento por sesión
        - Fatiga intra-serie
        - Distribución de potencia
        - Relaciones entre variables clave

        El código debe ejecutarse directamente y producir gráficos profesionales.
        """

        log.info(f"🧠 Iniciando análisis completo con {len(df)} registros...")
        
        start_time = time.time()
        
        response = ai_client.chat.completions.create(
            model=app.config["OPENAI_MODEL"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=4000,  # Suficiente para análisis + código
            response_format={"type": "json_object"}
        )
        
        elapsed_time = time.time() - start_time
        log.info(f"✅ IA respondió en {elapsed_time:.2f}s")
        
        result_text = response.choices[0].message.content
        
        # Validar JSON
        try:
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            log.error(f"❌ JSON inválido de IA, creando análisis básico")
            return create_complete_fallback_analysis(df)
            
    except Exception as e:
        log.error(f"❌ Error en IA: {str(e)}")
        return get_fallback_analysis(df)

def create_complete_fallback_analysis(df: pd.DataFrame) -> dict:
    """Crea análisis de respaldo usando TODOS los datos"""
    
    # Análisis básico con todos los datos
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Métricas básicas con todos los datos
    basic_stats = {}
    for col in numeric_cols:
        if col in df.columns:
            basic_stats[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()), 
                "promedio": float(df[col].mean()),
                "desviacion": float(df[col].std())
            }
    
    return {
        "analisis_completo": f"Análisis básico de {len(df)} registros completos. {len(numeric_cols)} variables numéricas analizadas.",
        "hallazgos_principales": [
            f"Dataset completo con {len(df)} mediciones",
            f"{len(numeric_cols)} variables cuantitativas disponibles",
            "Configure OPENAI_API_KEY para análisis avanzado con IA"
        ],
        "metricas_clave": basic_stats,
        "recomendaciones": [
            "Active análisis con IA para insights deportivos específicos",
            f"Se analizaron {len(df)} registros completos",
            "Los gráficos generados usan todos los datos disponibles"
        ],
        "codigo_graficos": generate_complete_chart_code(df, numeric_cols),
        "descripcion_graficos": "Gráficos básicos generados con todos los datos del dataset"
    }

def generate_complete_chart_code(df: pd.DataFrame, numeric_cols: list) -> str:
    """Genera código de gráficos usando TODOS los datos"""
    
    if len(numeric_cols) == 0:
        return "# No hay columnas numéricas para graficar"
    
    code = """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO

charts = []
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

try:
    # USANDO DATOS COMPLETOS - TODOS LOS REGISTROS
"""
    
    # Gráfico 1: Distribuciones de todas las columnas numéricas
    if len(numeric_cols) > 0:
        code += f"""
    # Distribución de variables numéricas (todos los datos)
    num_vars = {min(4, len(numeric_cols))}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, col in enumerate({numeric_cols}[:4]):
        if i < len(axes):
            axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'Distribución de {{col}} (n={{len(df)}})', fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf1 = BytesIO()
    plt.savefig(buf1, format='png', dpi=120, bbox_inches='tight')
    buf1.seek(0)
    charts.append(buf1)
    plt.close()
"""

    # Gráfico 2: Correlaciones si hay múltiples variables
    if len(numeric_cols) > 1:
        code += f"""
    # Mapa de correlaciones (todas las variables numéricas)
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_data = df[{numeric_cols}].select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
        ax.set_title('Correlaciones - Dataset Completo', fontweight='bold', pad=20)
        buf2 = BytesIO()
        plt.savefig(buf2, format='png', dpi=120, bbox_inches='tight')
        buf2.seek(0)
        charts.append(buf2)
        plt.close()
"""

    # Gráfico 3: Evolución temporal si hay columna de fecha/tiempo
    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'fecha', 'time', 'tiempo'])]
    if date_cols and len(numeric_cols) > 0:
        code += f"""
    # Evolución temporal (si hay datos de fecha)
    try:
        date_col = '{date_cols[0]}'
        if date_col in df.columns:
            # Intentar convertir a datetime
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            df_temp = df_temp.dropna(subset=[date_col])
            if len(df_temp) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_temp[date_col], df_temp['{numeric_cols[0]}'], marker='o', linewidth=2)
                ax.set_title('Evolución Temporal', fontweight='bold')
                ax.set_xlabel(date_col)
                ax.set_ylabel('{numeric_cols[0]}')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                buf3 = BytesIO()
                plt.savefig(buf3, format='png', dpi=120, bbox_inches='tight')
                buf3.seek(0)
                charts.append(buf3)
                plt.close()
    except Exception as e:
        print(f'Error en gráfico temporal: {{e}}')
"""

    code += "\nexcept Exception as e:\n    print(f'Error en gráficos: {e}')\n"
    
    return code

def get_fallback_analysis(df: pd.DataFrame) -> dict:
    """Análisis de respaldo cuando no hay IA"""
    return {
        "analisis_completo": f"Análisis básico de {len(df)} registros completos. Configure OPENAI_API_KEY para análisis avanzado con IA.",
        "hallazgos_principales": [
            f"Dataset procesado: {len(df)} registros",
            f"Variables disponibles: {list(df.columns)}",
            "Análisis IA no disponible"
        ],
        "metricas_clave": {
            "total_registros": len(df),
            "total_variables": len(df.columns),
            "configuracion": "Configure OPENAI_API_KEY"
        },
        "recomendaciones": [
            "Configure OPENAI_API_KEY para análisis deportivo con IA",
            f"Se procesaron {len(df)} registros completos",
            "Los gráficos usarán todos los datos disponibles"
        ],
        "codigo_graficos": generate_complete_chart_code(df, df.select_dtypes(include=['number']).columns.tolist()),
        "descripcion_graficos": "Gráficos básicos generados con dataset completo"
    }

# ==============================
# GENERACIÓN DE GRÁFICOS COMPLETOS
# ==============================

def generate_complete_charts(df: pd.DataFrame, python_code: str) -> List[BytesIO]:
    """
    Ejecuta código de gráficos usando TODOS los datos
    """
    charts = []
    
    if not python_code or "Configure OPENAI_API_KEY" in python_code:
        # Generar gráficos básicos con todos los datos
        return generate_basic_complete_charts(df)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        
        # Configuración profesional
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Variables disponibles - INCLUYENDO df COMPLETO
        local_vars = {
            'df': df,  # TODOS LOS DATOS
            'plt': plt,
            'sns': sns, 
            'np': np,
            'pd': pd,
            'BytesIO': BytesIO,
            'charts': []
        }
        
        # Ejecutar código generado por IA
        exec(python_code, local_vars)
        
        # Obtener gráficos
        charts = local_vars.get('charts', [])
        
        # Validar que sean BytesIO
        valid_charts = []
        for chart in charts:
            if isinstance(chart, BytesIO):
                valid_charts.append(chart)
        
        log.info(f"📊 {len(valid_charts)} gráficos personalizados generados")
        return valid_charts[:4]  # Máximo 4 gráficos
        
    except Exception as e:
        log.error(f"❌ Error ejecutando código de gráficos: {str(e)}")
        # Fallback: gráficos básicos con todos los datos
        return generate_basic_complete_charts(df)

def generate_basic_complete_charts(df: pd.DataFrame) -> List[BytesIO]:
    """Gráficos básicos usando TODOS los datos"""
    charts = []
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from io import BytesIO
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Gráfico 1: Distribución de la primera variable (todos los datos)
            fig, ax = plt.subplots(figsize=(10, 6))
            df[numeric_cols[0]].hist(ax=ax, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'Distribución de {numeric_cols[0]} (n={len(df)})', fontweight='bold')
            ax.set_xlabel(numeric_cols[0])
            ax.grid(True, alpha=0.3)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            charts.append(buf)
            plt.close()
        
        if len(numeric_cols) > 1:
            # Gráfico 2: Correlación entre primeras dos variables (todos los datos)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6, s=50)
            ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]} (n={len(df)})', fontweight='bold')
            ax.set_xlabel(numeric_cols[0])
            ax.set_ylabel(numeric_cols[1])
            ax.grid(True, alpha=0.3)
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
            buf.seek(0)
            charts.append(buf)
            plt.close()
            
    except Exception as e:
        log.error(f"Error en gráficos básicos completos: {e}")
    
    return charts

# ==============================
# GENERACIÓN DE PDF PROFESIONAL
# ==============================

def generate_professional_pdf(ai_result: dict, charts: List[BytesIO], meta: dict) -> str:
    """Genera PDF profesional con análisis completo"""
    
    try:
        pdf_path = os.path.join(app.config["UPLOAD_DIR"], f"reporte_{uuid.uuid4().hex}.pdf")
        
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        styles = getSampleStyleSheet()
        
        story = []
        
        # Header
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2E86AB'),
            spaceAfter=20,
            alignment=1,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("INFORME DE ANÁLISIS DEPORTIVO COMPLETO", title_style))
        
        # Información básica
        info_text = f"""
        <b>Entrenador:</b> {meta.get('nombre_entrenador', 'N/A')}<br/>
        <b>Atleta:</b> {meta.get('nombre_cliente', 'N/A')}<br/>
        <b>Fecha de análisis:</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Archivo analizado:</b> {meta.get('file_name', 'N/A')}<br/>
        <b>Registros procesados:</b> {meta.get('data_shape', 'N/A')}<br/>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Análisis Completo
        if 'analisis_completo' in ai_result:
            story.append(Paragraph("<b>📊 ANÁLISIS COMPLETO:</b>", styles['Normal']))
            analysis_text = str(ai_result['analisis_completo']).replace('\n', '<br/>')
            story.append(Paragraph(analysis_text, styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Hallazgos Principales
        if 'hallazgos_principales' in ai_result and ai_result['hallazgos_principales']:
            story.append(Paragraph("<b>🎯 HALLAZGOS PRINCIPALES:</b>", styles['Normal']))
            for hallazgo in ai_result['hallazgos_principales'][:5]:
                story.append(Paragraph(f"• {hallazgo}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Métricas Clave
        if 'metricas_clave' in ai_result:
            story.append(Paragraph("<b>📈 MÉTRICAS CLAVE:</b>", styles['Normal']))
            metrics = ai_result['metricas_clave']
            if isinstance(metrics, dict):
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        value_str = ", ".join([f"{k}: {v}" for k, v in value.items()])
                    else:
                        value_str = str(value)
                    story.append(Paragraph(f"<b>{key}:</b> {value_str}", styles['Normal']))
            story.append(Spacer(1, 15))
        
        # Gráficos
        if charts:
            story.append(Paragraph("<b>📊 ANÁLISIS VISUAL (DATOS COMPLETOS):</b>", styles['Normal']))
            story.append(Spacer(1, 10))
            
            for i, chart in enumerate(charts):
                try:
                    chart.seek(0)
                    img = ReportLabImage(chart, width=6*inch, height=4.5*inch)
                    story.append(img)
                    story.append(Spacer(1, 5))
                    
                    desc = f"Figura {i+1}: {ai_result.get('descripcion_graficos', 'Gráfico personalizado con todos los datos')}"
                    story.append(Paragraph(desc, styles['Italic']))
                    story.append(Spacer(1, 15))
                except Exception as e:
                    continue
        
        # Recomendaciones
        if 'recomendaciones' in ai_result and ai_result['recomendaciones']:
            story.append(Paragraph("<b>💡 RECOMENDACIONES PRÁCTICAS:</b>", styles['Normal']))
            for rec in ai_result['recomendaciones'][:6]:
                story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 20))
        footer_text = f"""
        <i>Reporte generado por InertiaX Pro - Análisis con IA<br/>
        Procesado el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}<br/>
        Todos los datos del dataset fueron utilizados en el análisis</i>
        """
        story.append(Paragraph(footer_text, styles['Italic']))
        
        doc.build(story)
        return pdf_path
        
    except Exception as e:
        log.error(f"Error generando PDF: {e}")
        # PDF de error mínimo
        error_path = os.path.join(app.config["UPLOAD_DIR"], f"error_{uuid.uuid4().hex}.pdf")
        c = canvas.Canvas(error_path)
        c.drawString(100, 750, "Error generando reporte completo")
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
    """Endpoint principal - usa TODOS los datos"""
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400
    
    f = request.files['file']
    if not f or f.filename == '':
        return jsonify({"error": "Archivo no especificado"}), 400
    
    try:
        # Validar extensión
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in app.config["ALLOWED_EXT"]:
            return jsonify({"error": f"Formato {ext} no soportado"}), 400
        
        # Configurar job
        job_id = _ensure_job()
        
        # Guardar archivo
        safe_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(_job_dir(job_id), safe_name)
        f.save(save_path)
        
        # Obtener datos del formulario
        nombre_entrenador = request.form.get('nombre_entrenador', 'Entrenador')
        nombre_cliente = request.form.get('nombre_cliente', 'Atleta')
        
        # Procesar archivo COMPLETO
        log.info("📁 Procesando dataset COMPLETO...")
        df = parse_dataframe(save_path)
        
        # Guardar metadata
        meta = {
            "file_name": f.filename,
            "file_path": save_path,
            "nombre_entrenador": nombre_entrenador,
            "nombre_cliente": nombre_cliente,
            "upload_time": datetime.now().isoformat(),
            "data_shape": f"{df.shape}",
            "total_registros": len(df)
        }
        _save_meta(job_id, meta)
        
        # Análisis con IA usando TODOS los datos
        log.info(f"🧠 Ejecutando análisis completo con {len(df)} registros...")
        ai_result = analyze_with_ai_complete(df, nombre_entrenador, nombre_cliente)
        
        # Generar gráficos personalizados con TODOS los datos
        log.info("📊 Generando gráficos personalizados...")
        charts = generate_complete_charts(df, ai_result.get('codigo_graficos', ''))
        
        # Generar PDF profesional
        log.info("📄 Creando reporte profesional...")
        pdf_path = generate_professional_pdf(ai_result, charts, meta)
        
        # Crear ZIP con resultados
        log.info("🗜️ Empacando resultados completos...")
        zip_path = os.path.join(_job_dir(job_id), f"reporte_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(pdf_path, "Reporte_Analisis_Completo.pdf")
            zf.write(save_path, f"datos_originales/{f.filename}")
            
            # Agregar análisis completo en JSON
            analysis_path = os.path.join(_job_dir(job_id), "analisis_completo.json")
            with open(analysis_path, "w", encoding="utf-8") as af:
                json.dump(ai_result, af, indent=2, ensure_ascii=False)
            zf.write(analysis_path, "analisis/resultados_completos.json")

        # Limpieza
        try:
            os.remove(pdf_path)
            os.remove(analysis_path)
        except:
            pass

        total_time = time.time() - start_time
        log.info(f"✅ Análisis COMPLETO finalizado en {total_time:.2f}s - {len(df)} registros procesados")
        
        return send_file(
            zip_path, 
            as_attachment=True, 
            download_name=f"Analisis_Completo_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
        )
        
    except Exception as e:
        log.error(f"❌ Error en procesamiento completo: {str(e)}")
        return jsonify({"error": f"Error procesando archivo: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "ok", 
        "message": "InertiaX Pro - Análisis Completo API",
        "openai_configured": ai_client is not None,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    log.info(f"🚀 INERTIAX PRO - ANÁLISIS COMPLETO - Puerto {port}")
    log.info(f"📊 Característica: Usa TODOS los datos del dataset")
    
    app.run(host="0.0.0.0", port=port, debug=False)
