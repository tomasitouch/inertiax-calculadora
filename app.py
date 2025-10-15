from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from docx import Document
from docx.shared import Inches
import mercadopago
from openai import OpenAI
import uuid
import os

app = Flask(__name__)
app.secret_key = "inertiax_secret_key"

# === Mercado Pago ===
mp = mercadopago.SDK(os.getenv("MP_ACCESS_TOKEN"))
DOMAIN_URL = "https://inertiax-calculadora-1.onrender.com"

# === OpenAI ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_file_path = None
form_data = {}
payment_approved = False


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

        unique_name = f"{uuid.uuid4()}.csv"
        current_file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(current_file_path)

        df = pd.read_csv(current_file_path)
        table_html = df.to_html(classes='table table-striped table-hover', index=False)

        return render_template(
            'index.html',
            table_html=table_html,
            filename=file.filename,
            form_data=form_data,
            show_payment=True
        )
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")


@app.route('/create_preference', methods=['POST'])
def create_preference():
    preference_data = {
        "items": [
            {
                "title": "Reporte InertiaX Editable",
                "quantity": 1,
                "unit_price": 2900,  # CLP
                "currency_id": "CLP"
            }
        ],
        "back_urls": {
            "success": f"{DOMAIN_URL}/success",
            "failure": f"{DOMAIN_URL}/cancel",
            "pending": f"{DOMAIN_URL}/pending"
        },
        "auto_return": "approved",
    }
    preference = mp.preference().create(preference_data)
    return jsonify(preference)


@app.route('/success')
def success():
    global payment_approved
    payment_approved = True
    return render_template('success.html')


@app.route('/cancel')
def cancel():
    return render_template('cancel.html')


@app.route('/generate_docx', methods=['GET'])
def generate_docx():
    global current_file_path, form_data, payment_approved
    try:
        if not payment_approved:
            return redirect(url_for('index'))

        df = pd.read_csv(current_file_path)

        resumen = f"Columnas: {list(df.columns)}. Ejemplo: {df.head(3).to_dict(orient='records')}."
        contexto = f"Tipo: {form_data.get('tipo_datos')}. Propósito: {form_data.get('proposito')}. Detalles: {form_data.get('detalles')}. Nombre: {form_data.get('nombre')}."

        prompt = f"""
        Eres un analista deportivo experto. Analiza el siguiente contexto y los datos:
        {contexto}
        {resumen}
        Genera un informe exhaustivo con métricas, tendencias, interpretación y recomendaciones.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis deportivo y redacción técnica profesional."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = completion.choices[0].message.content

        doc = Document()
        doc.add_heading("Reporte Profesional InertiaX", 0)
        doc.add_paragraph(f"Generado para: {form_data.get('nombre')}")
        doc.add_paragraph(f"Tipo de datos: {form_data.get('tipo_datos')}")
        doc.add_paragraph(f"Propósito: {form_data.get('proposito')}")
        doc.add_paragraph(f"Detalles: {form_data.get('detalles')}")
        doc.add_paragraph("\n")
        doc.add_heading("Análisis detallado", level=1)
        doc.add_paragraph(analysis)

        doc.add_page_break()
        doc.add_heading("Vista previa de datos", level=1)
        table = doc.add_table(rows=1, cols=len(df.columns))
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(df.columns):
            hdr_cells[i].text = col
        for _, row in df.head(10).iterrows():
            row_cells = table.add_row().cells
            for i, val in enumerate(row):
                row_cells[i].text = str(val)

        output_path = os.path.join(UPLOAD_FOLDER, "reporte_inertiax.docx")
        doc.save(output_path)

        return send_file(output_path, as_attachment=True, download_name="reporte_inertiax.docx")

    except Exception as e:
        return f"Error al generar el documento: {e}", 500
