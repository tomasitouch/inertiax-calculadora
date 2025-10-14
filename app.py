import os
import pandas as pd
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI
from docx import Document
from docx.shared import Pt, Inches
from datetime import datetime

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ✅ Cliente OpenRouter optimizado
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "InertiaX Analyzer activo 🚀"})

@app.route("/analizar", methods=["POST"])
def analizar():
    if "archivo" not in request.files:
        return jsonify({"error": "No se subió ningún archivo CSV"}), 400

    archivo = request.files["archivo"]
    filepath = os.path.join(UPLOAD_DIR, archivo.filename)
    archivo.save(filepath)

    try:
        # 🧮 Procesamiento rápido con Pandas (resumen estadístico)
        df = pd.read_csv(filepath)
        resumen = df.describe(include="all").round(3).to_string()

        # 💬 Prompt optimizado para rapidez y consistencia
        prompt = f"""
        Actúa como un analista deportivo profesional.
        Analiza el siguiente resumen estadístico de un entrenamiento:
        {resumen}

        Entrega un reporte estructurado con:
        1️⃣ Resumen general
        2️⃣ Fortalezas 💪
        3️⃣ Debilidades ⚠️
        4️⃣ Sugerencias 🎯

        Sé conciso, claro y técnico. No repitas encabezados, solo texto en cada sección.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        texto = completion.choices[0].message["content"]

        # 🧾 Crear documento Word profesional
        doc = Document()

        # Cabecera
        header = doc.sections[0].header
        header_para = header.paragraphs[0]
        header_para.text = "InertiaX Analyzer – Reporte de Rendimiento"
        header_para.style = doc.styles["Header"]

        doc.add_heading("📊 Reporte de Rendimiento – InertiaX Analyzer", level=1)
        doc.add_paragraph(f"Generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}").italic = True
        doc.add_paragraph(" ")

        # Añadir contenido del reporte
        for line in texto.split("\n"):
            if line.strip():
                p = doc.add_paragraph(line.strip())
                p.style = doc.styles["Normal"]
                for run in p.runs:
                    run.font.size = Pt(11)

        # Pie de página
        footer = doc.sections[0].footer.paragraphs[0]
        footer.text = "© InertiaX SpA | Generado automáticamente por InertiaX Analyzer"

        # Guardar archivo
        filename = f"Reporte_InertiaX_{os.path.splitext(archivo.filename)[0]}.docx"
        output_path = os.path.join(REPORT_DIR, filename)
        doc.save(output_path)

        return send_file(output_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
