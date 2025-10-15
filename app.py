from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
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
current_file_path = None  # Guardamos el último archivo procesado


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global current_file_path
    try:
        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        unique_name = f"{uuid.uuid4()}.csv"
        current_file_path = os.path.join(UPLOAD_FOLDER, unique_name)
        file.save(current_file_path)

        df = pd.read_csv(current_file_path)
        table_html = df.to_html(classes='table table-striped table-hover', index=False)
        return render_template('index.html', table_html=table_html, filename=file.filename)
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")


@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    global current_file_path
    try:
        if not current_file_path or not os.path.exists(current_file_path):
            return "No hay datos cargados.", 400

        df = pd.read_csv(current_file_path)

        # === IA: análisis textual ===
        csv_summary = f"Columnas: {list(df.columns)}. Primeras filas: {df.head(3).to_dict(orient='records')}."
        prompt = f"Analiza los siguientes datos de rendimiento deportivo y genera un informe breve con observaciones y conclusiones.\n{csv_summary}"

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de datos deportivos."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = completion.choices[0].message.content

        # === Gráficos ===
        img_bufs = []
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col].dropna(), kde=True, color='#007bff')
                ax.set_title(f"Distribución de {col}")
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_bufs.append(buf)
                plt.close()

            # Correlaciones
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(corr, annot=True, cmap="coolwarm")
                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                img_bufs.append(buf)
                plt.close()

        # === Generar PDF ===
        pdf_buffer = BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(60, 760, "📊 Reporte de Análisis InertiaX")
        c.setFont("Helvetica", 11)
        c.drawString(60, 740, "Análisis automático de datos de rendimiento deportivo")
        c.line(60, 735, 540, 735)

        text = c.beginText(60, 710)
        text.setFont("Helvetica", 10)
        for line in analysis.split('\n'):
            text.textLine(line)
        c.drawText(text)

        y = 400
        for buf in img_bufs:
            if y < 200:
                c.showPage()
                y = 700
            img = ImageReader(buf)
            c.drawImage(img, 60, y - 200, width=480, height=180)
            y -= 220

        c.save()
        pdf_buffer.seek(0)

        return send_file(BytesIO(pdf_buffer.getvalue()),
                         as_attachment=False,
                         download_name="reporte_inertiax.pdf",
                         mimetype="application/pdf")

    except Exception as e:
        return f"Error al generar PDF: {e}", 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
