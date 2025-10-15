from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from docx import Document
from docx.shared import Inches
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps
import requests
import mercadopago
from openai import OpenAI
import zipfile
import os, uuid

# ------------------ Config ------------------
app = Flask(__name__)
app.secret_key = "inertiax_secret_key"

# Mercado Pago
mp = mercadopago.SDK(os.getenv("MP_ACCESS_TOKEN"))
DOMAIN_URL = "https://inertiax-calculadora-1.onrender.com"  # ajusta si usas otro dominio

# OpenAI via OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

# Paths y estado simple
UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_file_path = None
form_data = {}
payment_approved = False

# Logo circular (descargado desde tu URL)
LOGO_URL = "https://scontent-scl3-1.cdninstagram.com/v/v/t51.2885-19/523933037_17845198551536603_8934147041556657694_n.jpg?efg=eyJ2ZW5jb2RlX3RhZyI6InByb2ZpbGVfcGljLmRqYW5nby4xMDgwLmMyIn0&_nc_ht=scontent-scl3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QHOy6rgcBc7EW5ZszSp4lJTdyOpbiDr73ZBLQu3R0fFLrhnThZGWbbGejuqVpYJ9a4&_nc_ohc=hYgEXbr2xVoQ7kNvwGzhiGQ&_nc_gid=HUhI8RtTJJyKdKSj0v0qOQ&edm=AP4sbd4BAAAA&ccb=7-5&oh=00_Afd9hP8K3ACP2osMWfw_db9f5k_-SUItXzjPX0kUjnd79A&oe=68F4E13F&_nc_sid=7a9f4b"

# ------------------ Utilidades ------------------
def fetch_circular_logo(url: str, size_px: int = 180) -> BytesIO:
    """Descarga el logo, lo recorta circularmente y devuelve un PNG en memoria."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    img = ImageOps.fit(img, (size_px, size_px), method=Image.LANCZOS, centering=(0.5, 0.5))
    mask = Image.new("L", (size_px, size_px), 0)
    mask_draw = Image.new("L", (size_px, size_px), 0)
    mask = Image.new("L", (size_px, size_px), 0)
    # círculo
    circle = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw = ImageDraw if 'ImageDraw' in globals() else __import__('PIL.ImageDraw').ImageDraw
    d = ImageDraw.Draw(circle)
    d.ellipse((0, 0, size_px, size_px), fill=255)
    img.putalpha(circle)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def parse_dataframe(path: str) -> pd.DataFrame:
    """Lee CSV o Excel según extensión."""
    name = os.path.basename(path).lower()
    if name.endswith(".csv"):
        return pd.read_csv(path)
    else:
        # Excel: soporta .xlsx, .xls
        return pd.read_excel(path)

def guess_datetime_column(df: pd.DataFrame):
    """Intenta detectar una columna de fecha/tiempo para series temporales."""
    candidates = [c for c in df.columns if any(k in c.lower() for k in ["date", "fecha", "time", "día", "dia"])]
    for c in candidates:
        try:
            dt = pd.to_datetime(df[c])
            return c, dt
        except Exception:
            continue
    return None, None

def generate_figures(df: pd.DataFrame):
    """Genera figuras relevantes en base a los datos numéricos y temporalidad."""
    figures = []
    sns.set_theme(style="whitegrid")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Histogramas (hasta 4)
    for col in num_cols[:4]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        figures.append(buf)

    # Boxplot multivariable
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df[num_cols], ax=ax)
        ax.set_title("Distribución por variable (Boxplot)")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        figures.append(buf)

    # Correlaciones
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Matriz de correlaciones")
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png", dpi=180)
        plt.close(fig)
        buf.seek(0)
        figures.append(buf)

    # Serie temporal (si hay columna datetime)
    date_col, parsed = guess_datetime_column(df)
    if date_col is not None:
        # Elegimos la primera numérica para ilustrar tendencia
        if num_cols:
            try:
                ts = df.copy()
                ts[date_col] = pd.to_datetime(ts[date_col])
                ts = ts.sort_values(date_col)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(ts[date_col], ts[num_cols[0]], marker="o", linewidth=1)
                ax.set_title(f"Tendencia temporal de {num_cols[0]} vs {date_col}")
                ax.set_xlabel(date_col)
                ax.set_ylabel(num_cols[0])
                buf = BytesIO()
                fig.tight_layout()
                fig.savefig(buf, format="png", dpi=180)
                plt.close(fig)
                buf.seek(0)
                figures.append(buf)
            except Exception:
                pass

    return figures

def wrap_canvas_text(c: canvas.Canvas, text: str, x: int, y: int, width: int, leading: int = 14, font="Helvetica", size=10):
    """Dibuja texto con saltos de línea automáticos en ReportLab."""
    from textwrap import wrap
    c.setFont(font, size)
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")  # línea en blanco
        else:
            lines += wrap(paragraph, width=width)
    cursor_y = y
    for line in lines:
        c.drawString(x, cursor_y, line)
        cursor_y -= leading
        if cursor_y < 80:  # nueva página
            c.showPage()
            c.setFont(font, size)
            cursor_y = 750
    return cursor_y

def generate_pdf(analysis_text: str, df: pd.DataFrame, logo_buf: BytesIO, meta: dict) -> str:
    """Crea un PDF profesional con logo circular y gráficos."""
    pdf_path = os.path.join(UPLOAD_FOLDER, f"reporte_inertiax_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Encabezado con logo
    try:
        img = ImageReader(logo_buf)
        c.drawImage(img, 40, 700, width=80, height=80, mask='auto')
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(140, 760, "Reporte Profesional InertiaX")
    c.setFont("Helvetica", 10)
    c.drawString(140, 742, f"Deportista/Entrenador: {meta.get('nombre') or '-'}")
    c.drawString(140, 728, f"Tipo de datos: {meta.get('tipo_datos') or '-'}")
    c.drawString(140, 714, f"Propósito: {meta.get('proposito') or '-'}")

    # Texto analítico
    wrap_canvas_text(c, analysis_text, x=40, y=680, width=90)  # 90 ~ caracteres por línea aprox

    # Gráficos
    figs = generate_figures(df)
    y = 650
    for fb in figs:
        if y < 220:
            c.showPage()
            y = 730
        try:
            img = ImageReader(fb)
            c.drawImage(img, 60, y-200, width=480, height=180)
            y -= 220
        except Exception:
            continue

    c.save()
    return pdf_path

def generate_docx(analysis_text: str, df: pd.DataFrame, logo_buf: BytesIO, meta: dict) -> str:
    """Crea un Word editable con logo circular, análisis y tablas/gráficos."""
    doc = Document()
    # Logo
    try:
        doc.add_picture(logo_buf, width=Inches(1.2))
    except Exception:
        pass

    doc.add_heading("Reporte Profesional InertiaX", 0)
    doc.add_paragraph(f"Generado para: {meta.get('nombre') or '-'}")
    doc.add_paragraph(f"Tipo de datos: {meta.get('tipo_datos') or '-'}")
    doc.add_paragraph(f"Propósito: {meta.get('proposito') or '-'}")
    if meta.get('detalles'):
        doc.add_paragraph(f"Detalles: {meta.get('detalles')}")
    doc.add_paragraph(" ")

    doc.add_heading("Análisis detallado", level=1)
    for p in analysis_text.split("\n"):
        doc.add_paragraph(p)

    doc.add_page_break()
    doc.add_heading("Vista previa de datos", level=1)

    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr[i].text = str(col)
    for _, row in df.head(15).iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    # Gráficos
    figs = generate_figures(df)
    for fb in figs:
        doc.add_page_break()
        doc.add_picture(fb, width=Inches(6.0))

    out_path = os.path.join(UPLOAD_FOLDER, f"reporte_inertiax_{uuid.uuid4().hex}.docx")
    doc.save(out_path)
    return out_path

def run_ai_analysis(df: pd.DataFrame, meta: dict) -> str:
    """Llama al modelo para análisis profundo y estructurado."""
    resumen = f"Columnas: {list(df.columns)}. Primeras filas: {df.head(3).to_dict(orient='records')}."
    contexto = f"Tipo: {meta.get('tipo_datos')}. Propósito: {meta.get('proposito')}. Detalles: {meta.get('detalles')}. Nombre: {meta.get('nombre')}."
    system_prompt = """
Eres un analista experto en rendimiento deportivo, biomecánica y ciencia del entrenamiento.
Analiza datos crudos (fuerza, velocidad, potencia, salto, tiempos, cargas, etc.).
Entrega un informe profesional con secciones:
1) Resumen ejecutivo,
2) Análisis estadístico (medias, desviaciones, outliers, correlaciones, tendencias),
3) Interpretación fisiológica/biomecánica,
4) Recomendaciones prácticas y accionables,
5) Limitaciones y próximos pasos.
Usa tono técnico claro para entrenadores/deportistas. Sé específico y evita vaguedades.
"""
    user_prompt = f"""
Contexto: {contexto}
Datos (muestra): {resumen}
Genera el informe siguiendo la estructura, referenciando variables y métricas concretas que veas pertinentes.
"""

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )
    return completion.choices[0].message.content

# ------------------ Rutas ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global current_file_path, form_data, payment_approved
    payment_approved = False
    try:
        form_data = {
            "tipo_datos": request.form.get("tipo_datos"),
            "proposito": request.form.get("proposito"),
            "detalles": request.form.get("detalles"),
            "nombre": request.form.get("nombre")
        }

        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        # Guardar con extensión original
        ext = os.path.splitext(file.filename)[1] or ".csv"
        unique_name = f"{uuid.uuid4()}{ext}"
        current_file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(current_file_path)

        df = parse_dataframe(current_file_path)
        table_html = df.to_html(classes='table table-striped table-hover', index=False)

        return render_template('index.html',
                               table_html=table_html,
                               filename=file.filename,
                               form_data=form_data,
                               show_payment=True)
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")

@app.route('/create_preference', methods=['POST'])
def create_preference():
    # Precio fijo o dinámico por propósito (ajusta a gusto)
    price = 2900
    prop = (form_data.get("proposito") or "").lower()
    if "estadístico" in prop:
        price = 3900
    elif "avanzado" in prop:
        price = 5900
    elif "todo" in prop:
        price = 6900

    preference_data = {
        "items": [{
            "title": f"Reporte InertiaX - {form_data.get('proposito') or 'Análisis'}",
            "quantity": 1,
            "unit_price": price,  # CLP
            "currency_id": "CLP"
        }],
        "back_urls": {
            "success": f"{DOMAIN_URL}/success",
            "failure": f"{DOMAIN_URL}/cancel",
            "pending": f"{DOMAIN_URL}/cancel"
        },
        "auto_return": "approved",
    }
    preference = mp.preference().create(preference_data)
    # Importante: devolver el objeto "response" para obtener "id"
    return jsonify(preference["response"])

@app.route('/success')
def success():
    global payment_approved
    payment_approved = True
    return render_template('success.html')

@app.route('/cancel')
def cancel():
    return render_template('cancel.html')

@app.route('/download_bundle')
def download_bundle():
    """Tras el pago, genera análisis, DOCX, PDF y devuelve un ZIP con ambos."""
    global payment_approved, current_file_path, form_data
    try:
        if not payment_approved:
            return redirect(url_for('index'))
        if not current_file_path or not os.path.exists(current_file_path):
            return "No hay datos cargados.", 400

        df = parse_dataframe(current_file_path)
        analysis = run_ai_analysis(df, form_data)

        # Logo circular
        try:
            logo_buf = fetch_circular_logo(LOGO_URL, size_px=220)
        except Exception:
            logo_buf = BytesIO()  # sin logo si falla

        # Generar DOCX y PDF
        docx_path = generate_docx(analysis, df, logo_buf, form_data)
        pdf_path  = generate_pdf(analysis, df, logo_buf, form_data)

        # Empaquetar ZIP
        zip_path = os.path.join(UPLOAD_FOLDER, f"reporte_inertiax_{uuid.uuid4().hex}.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(docx_path, arcname="reporte_inertiax.docx")
            zf.write(pdf_path,  arcname="reporte_inertiax.pdf")

        return send_file(zip_path, as_attachment=True, download_name="inertiax_reporte.zip")

    except Exception as e:
        return f"Error al generar y descargar: {e}", 500

# ------------------ Run ------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
