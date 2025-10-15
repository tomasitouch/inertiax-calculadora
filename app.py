from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from docx import Document
from docx.shared import Inches
import os
from openai import OpenAI
import uuid

app = Flask(__name__)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

UPLOAD_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_file_path = None
form_data = {}  # Aquí guardaremos los datos del formulario


@app.route('/')
def index():
    return render_template('index.html')


# -------- FORMULARIO + SUBIDA --------
@app.route('/upload', methods=['POST'])
def upload():
    global current_file_path, form_data
    try:
        # Datos del formulario
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
            form_data=form_data
        )
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")


# -------- GENERAR ANALISIS PROFUNDO Y DOCX --------
@app.route('/generate_docx', methods=['GET'])
def generate_docx():
    global current_file_path, form_data
    try:
        if not current_file_path or not os.path.exists(current_file_path):
            return "No hay datos cargados.", 400

        df = pd.read_csv(current_file_path)

        # === Análisis con IA según el propósito ===
        resumen = f"Columnas: {list(df.columns)}. Ejemplo de filas: {df.head(3).to_dict(orient='records')}."
        contexto = (
            f"Tipo de datos: {form_data.get('tipo_datos')}. "
            f"Propósito: {form_data.get('proposito')}. "
            f"Detalles: {form_data.get('detalles')}. "
            f"Nombre asociado: {form_data.get('nombre')}."
        )

        prompt = f"""
        Eres un analista deportivo experto. A partir del siguiente contexto:
        {contexto}

        Y con los datos cargados:
        {resumen}

        Genera un informe en formato textual detallado que incluya:
        - Introducción con descripción del tipo de análisis.
        - Principales hallazgos y métricas clave (valores medios, desviaciones, tendencias).
        - Análisis estadístico e interpretación de los resultados.
        - Recomendaciones o conclusiones personalizadas para el caso indicado.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis deportivo y redacción técnica profesional."},
                {"role": "user", "content": prompt}
            ]
        )

        analysis = completion.choices[0].message.content

        # --- Crear Word ---
        doc = Document()
        doc.add_heading("Reporte InertiaX", 0)
        doc.add_paragraph(f"Generado para: {form_data.get('nombre')}")
        doc.add_paragraph(f"Tipo de datos: {form_data.get('tipo_datos')}")
        doc.add_paragraph(f"Propósito: {form_data.get('proposito')}")
        doc.add_paragraph(f"Detalles: {form_data.get('detalles')}")
        doc.add_paragraph("\n")

        doc.add_heading("Análisis detallado", level=1)
        doc.add_paragraph(analysis)

        doc.add_page_break()

        doc.add_heading("Datos generales", level=1)
        table = doc.add_table(rows=1, cols=len(df.columns))
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(df.columns):
            hdr_cells[i].text = col

        for _, row in df.head(10).iterrows():  # Solo primeras 10 filas
            row_cells = table.add_row().cells
            for i, val in enumerate(row):
                row_cells[i].text = str(val)

        # --- Gráfico ejemplo ---
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) > 0:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df[num_cols])
            plt.title("Distribución de variables numéricas")
            graph_path = os.path.join(UPLOAD_FOLDER, "temp_plot.png")
            plt.savefig(graph_path)
            plt.close()
            doc.add_picture(graph_path, width=Inches(5.5))

        output_path = os.path.join(UPLOAD_FOLDER, "reporte_inertiax.docx")
        doc.save(output_path)

        return send_file(output_path, as_attachment=True, download_name="reporte_inertiax.docx")

    except Exception as e:
        return f"Error al generar el documento: {e}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
