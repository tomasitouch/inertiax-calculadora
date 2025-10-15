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
from PIL import Image, ImageOps, ImageDraw
import requests
import mercadopago
from openai import OpenAI
import zipfile
import os, uuid

# ------------------ CONFIG ------------------
app = Flask(__name__)
app.secret_key = "inertiax_secret_key"

# Mercado Pago
mp = mercadopago.SDK(os.getenv("MP_ACCESS_TOKEN"))
DOMAIN_URL = "https://inertiax-calculadora-1.onrender.com"

# OpenAI via OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

# Directorios temporales
UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

current_file_path = None
form_data = {}
payment_approved = False

# Códigos de invitado válidos
CODIGOS_INVITADO = ["INERTIAXVIP2025", "ENTRENADORPRO", "INVEXORTEST"]

# Logo circular
LOGO_URL = "https://scontent-scl3-1.cdninstagram.com/v/t51.2885-19/523933037_17845198551536603_8934147041556657694_n.jpg?efg=eyJ2ZW5jb2RlX3RhZyI6InByb2ZpbGVfcGljLmRqYW5nby4xMDgwLmMyIn0&_nc_ht=scontent-scl3-1.cdninstagram.com&_nc_cat=111&_nc_oc=Q6cZ2QHOy6rgcBc7EW5ZszSp4lJTdyOpbiDr73ZBLQu3R0fFLrhnThZGWbbGejuqVpYJ9a4&_nc_ohc=hYgEXbr2xVoQ7kNvwGzhiGQ&_nc_gid=HUhI8RtTJJyKdKSj0v0qOQ&edm=AP4sbd4BAAAA&ccb=7-5&oh=00_Afd9hP8K3ACP2osMWfw_db9f5k_-SUItXzjPX0kUjnd79A&oe=68F4E13F&_nc_sid=7a9f4b"


# ------------------ FUNCIONES ------------------
def fetch_circular_logo(url: str, size_px: int = 180) -> BytesIO:
    """Descarga el logo y lo recorta circularmente."""
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGBA")
    img = ImageOps.fit(img, (size_px, size_px), method=Image.LANCZOS)
    mask = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size_px, size_px), fill=255)
    img.putalpha(mask)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def parse_dataframe(path: str) -> pd.DataFrame:
    """Lee CSV o Excel automáticamente."""
    name = os.path.basename(path).lower()
    if name.endswith(".csv"):
        return pd.read_csv(path)
    else:
        return pd.read_excel(path)


def guess_datetime_column(df: pd.DataFrame):
    """Detecta columna temporal para análisis de series."""
    for c in df.columns:
        if any(k in c.lower() for k in ["date", "fecha", "time", "día", "dia"]):
            try:
                return c, pd.to_datetime(df[c])
            except Exception:
                continue
    return None, None


def generate_figures(df: pd.DataFrame):
    """Genera gráficos automáticos con seaborn."""
    sns.set_theme(style="whitegrid")
    figs = []
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in num_cols[:3]:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribución de {col}")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)

    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df[num_cols], ax=ax)
        ax.set_title("Comparación por variable")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)

    date_col, parsed = guess_datetime_column(df)
    if date_col and num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_sorted = df.sort_values(by=date_col)
        ax.plot(df_sorted[date_col], df_sorted[num_cols[0]], linewidth=1.5)
        ax.set_title(f"Tendencia temporal de {num_cols[0]}")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        figs.append(buf)
    return figs


def generate_pdf(analysis_text, df, logo_buf, meta):
    pdf_path = os.path.join(UPLOAD_FOLDER, f"reporte_{uuid.uuid4().hex}.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)

    # Logo
    try:
        c.drawImage(ImageReader(logo_buf), 40, 700, width=80, height=80, mask='auto')
    except Exception:
        pass

    c.setFont("Helvetica-Bold", 16)
    c.drawString(150, 760, "Reporte Profesional InertiaX")
    c.setFont("Helvetica", 10)
    c.drawString(150, 742, f"Nombre: {meta.get('nombre')}")
    c.drawString(150, 728, f"Tipo de datos: {meta.get('tipo_datos')}")
    c.drawString(150, 714, f"Propósito: {meta.get('proposito')}")

    text_object = c.beginText(50, 690)
    text_object.setFont("Helvetica", 10)
    for line in analysis_text.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)

    for fig in generate_figures(df):
        c.showPage()
        c.drawImage(ImageReader(fig), 50, 250, width=500, height=350)

    c.save()
    return pdf_path


def generate_docx(analysis_text, df, logo_buf, meta):
    doc = Document()
    try:
        doc.add_picture(logo_buf, width=Inches(1.2))
    except Exception:
        pass

    doc.add_heading("Reporte Profesional InertiaX", 0)
    doc.add_paragraph(f"Nombre: {meta.get('nombre')}")
    doc.add_paragraph(f"Tipo de datos: {meta.get('tipo_datos')}")
    doc.add_paragraph(f"Propósito: {meta.get('proposito')}")
    if meta.get('detalles'):
        doc.add_paragraph(f"Detalles: {meta.get('detalles')}")

    doc.add_heading("Análisis IA", level=1)
    doc.add_paragraph(analysis_text)

    doc.add_heading("Vista previa de datos", level=1)
    table = doc.add_table(rows=1, cols=len(df.columns))
    hdr_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        hdr_cells[i].text = col
    for _, row in df.head(10).iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)

    for fig in generate_figures(df):
        doc.add_page_break()
        doc.add_picture(fig, width=Inches(5.5))

    path = os.path.join(UPLOAD_FOLDER, f"reporte_{uuid.uuid4().hex}.docx")
    doc.save(path)
    return path


def run_ai_analysis(df, meta):
    resumen = f"Columnas: {list(df.columns)}. Ejemplo: {df.head(3).to_dict(orient='records')}."
    contexto = f"Tipo: {meta.get('tipo_datos')}. Propósito: {meta.get('proposito')}. Detalles: {meta.get('detalles')}."
    prompt = f"""
Eres un analista experto en rendimiento deportivo. Analiza los datos y entrega:
1. Resumen general
2. Análisis estadístico
3. Interpretación fisiológica
4. Recomendaciones prácticas
Contexto: {contexto}
Datos: {resumen}
"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un experto en biomecánica y entrenamiento deportivo."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


# ------------------ RUTAS ------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global current_file_path, form_data, payment_approved
    payment_approved = False
    mensaje = None

    try:
        form_data = {
            "tipo_datos": request.form.get("tipo_datos"),
            "proposito": request.form.get("proposito"),
            "detalles": request.form.get("detalles"),
            "nombre": request.form.get("nombre"),
        }

        codigo = request.form.get("codigo_invitado", "").strip()
        if codigo in CODIGOS_INVITADO:
            payment_approved = True
            mensaje = "🔓 Código de invitado válido. Puedes generar tu reporte sin pagar."

        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        ext = os.path.splitext(file.filename)[1] or ".csv"
        filename = f"{uuid.uuid4()}{ext}"
        current_file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(current_file_path)

        df = parse_dataframe(current_file_path)
        table_html = df.to_html(classes='table table-striped table-hover', index=False)

        return render_template(
            'index.html',
            table_html=table_html,
            filename=file.filename,
            form_data=form_data,
            mensaje=mensaje,
            show_payment=not payment_approved
        )

    except Exception as e:
        return render_template('index.html', error=f"Error: {e}")


@app.route('/create_preference', methods=['POST'])
def create_preference():
    preference_data = {
        "items": [{"title": "Reporte InertiaX", "quantity": 1, "unit_price": 2900, "currency_id": "CLP"}],
        "back_urls": {"success": f"{DOMAIN_URL}/success", "failure": f"{DOMAIN_URL}/cancel", "pending": f"{DOMAIN_URL}/cancel"},
        "auto_return": "approved",
    }
    pref = mp.preference().create(preference_data)
    return jsonify(pref["response"])


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
    global payment_approved, current_file_path, form_data
    if not payment_approved:
        return redirect(url_for('index'))

    df = parse_dataframe(current_file_path)
    analysis = run_ai_analysis(df, form_data)

    try:
        logo_buf = fetch_circular_logo(LOGO_URL, size_px=220)
    except Exception:
        logo_buf = BytesIO()

    docx = generate_docx(analysis, df, logo_buf, form_data)
    pdf = generate_pdf(analysis, df, logo_buf, form_data)

    zip_path = os.path.join(UPLOAD_FOLDER, f"reporte_{uuid.uuid4().hex}.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(docx, "reporte_inertiax.docx")
        zf.write(pdf, "reporte_inertiax.pdf")

    return send_file(zip_path, as_attachment=True, download_name="reporte_inertiax.zip")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
