from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
import os
from openai import OpenAI

app = Flask(__name__)
app.secret_key = "inertiax_secret_key"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

# ------------------- HOME -------------------
@app.route('/')
def index():
    return render_template('index.html')

# ------------------- SUBIR CSV -------------------
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if not file:
            return "No se subió ningún archivo.", 400

        df = pd.read_csv(file)
        session['csv_data'] = df.to_json(orient='records')
        table_html = df.to_html(classes='table table-striped table-hover', index=False)
        return render_template('index.html', table_html=table_html, filename=file.filename)
    except Exception as e:
        return render_template('index.html', error=f"Error al procesar el archivo: {e}")

# ------------------- CHAT IA -------------------
@app.route('/chat', methods=['POST'])
def chat():
    try:
        msg = request.json.get("message", "")
        if not msg:
            return jsonify({"response": "Por favor escribe algo."})

        csv_json = session.get('csv_data', None)
        csv_summary = ""

        if csv_json:
            df = pd.read_json(csv_json)
            csv_summary = f"Columnas: {list(df.columns)}. Primeras filas: {df.head(3).to_dict(orient='records')}."

        prompt = f"Eres un analista deportivo experto. Analiza los siguientes datos y responde brevemente:\n{csv_summary}\nUsuario: {msg}"

        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un analista de datos deportivos experto en rendimiento físico y biomecánica."},
                {"role": "user", "content": prompt}
            ]
        )

        reply = res.choices[0].message.content
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error: {e}"}), 500

# ------------------- GENERAR PDF CON GRÁFICOS -------------------
@app.route('/generate_pdf', methods=['GET'])
def generate_pdf():
    try:
        csv_json = session.get('csv_data', None)
        if not csv_json:
            return "No hay datos cargados", 400

        df = pd.read_json(csv_json)

        # --- 1. Análisis IA ---
        csv_summary = f"Columnas: {list(df.columns)}. Primeras filas: {df.head(3).to_dict(orient='records')}."
        prompt = f"Analiza los siguientes datos deportivos y genera un informe profesional con observaciones, tendencias y conclusiones.\n{csv_summary}"

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de datos deportivos."},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = completion.choices[0].message.content

        # --- 2. Generar gráficos ---
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

        # --- 3. Crear PDF ---
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
        session['last_pdf'] = pdf_buffer.getvalue()

        return send_file(BytesIO(pdf_buffer.getvalue()),
                         as_attachment=False,
                         download_name="reporte_inertiax.pdf",
                         mimetype="application/pdf")

    except Exception as e:
        return f"Error generando PDF: {e}", 500

@app.route('/download_pdf')
def download_pdf():
    if 'last_pdf' not in session:
        return "No hay PDF generado", 400
    return send_file(BytesIO(session['last_pdf']),
                     as_attachment=True,
                     download_name="reporte_inertiax.pdf",
                     mimetype="application/pdf")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
