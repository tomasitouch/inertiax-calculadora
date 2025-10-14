import os
import pandas as pd
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from docx import Document
from docx.shared import Pt

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["REPORTS_FOLDER"] = "reports"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["REPORTS_FOLDER"], exist_ok=True)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

@app.route("/analizar", methods=["POST"])
def analizar():
    if "archivo" not in request.files:
        return jsonify({"error": "No se subió ningún archivo"}), 400

    file = request.files["archivo"]
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Leer el CSV
    df = pd.read_csv(filepath)
    resumen = df.describe(include='all').to_string()

    # Generar el prompt para la IA
    prompt = f"""
    Eres un analista de rendimiento deportivo especializado en fuerza y velocidad.
    Analiza las siguientes estadísticas del entrenamiento:
    {resumen}

    Genera un reporte estructurado con los siguientes apartados:
    1. Resumen general
    2. Fortalezas 💪
    3. Debilidades ⚠️
    4. Sugerencias 🎯

    Usa lenguaje profesional, claro y con enfoque en optimización del rendimiento.
    """

    # Generar el texto del reporte con IA
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    reporte = completion.choices[0].message["content"]

    # Crear el documento Word
    doc = Document()
    doc.add_heading("Reporte de Rendimiento - InertiaX Analyzer", level=1)
    doc.add_paragraph(" ")

    for seccion in ["Resumen general", "Fortalezas", "Debilidades", "Sugerencias"]:
        if seccion.lower() in reporte.lower():
            doc.add_heading(seccion, level=2)
        # Dividir el texto y agregarlo en párrafos legibles
        for linea in reporte.split("\n"):
            if linea.strip():
                run = doc.add_paragraph(linea.strip()).runs[0]
                run.font.size = Pt(11)

    # Guardar el archivo Word
    filename = f"Reporte_InertiaX_{os.path.splitext(file.filename)[0]}.docx"
    report_path = os.path.join(app.config["REPORTS_FOLDER"], filename)
    doc.save(report_path)

    # Devolver el archivo para descarga directa
    return send_file(report_path, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
