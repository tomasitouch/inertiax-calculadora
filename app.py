from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from openai import OpenAI
from docx import Document
import pandas as pd
import os
import io

app = Flask(__name__)
CORS(app)

# Configura OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

@app.route("/")
def home():
    return "InertiaX Analyzer activo 🚀"

@app.route("/analizar", methods=["POST"])
def analizar():
    try:
        # === 1. Leer archivo CSV ===
        archivo = request.files["archivo"]
        df = pd.read_csv(archivo, sep=None, engine="python")  # autodetecta el separador

        resumen = df.describe().to_string()

        # === 2. Pedir análisis a la IA ===
        prompt = f"""
        Eres un analista deportivo. Analiza los siguientes datos de rendimiento y
        entrega un resumen técnico de los resultados, fortalezas, debilidades y
        sugerencias de mejora para el atleta.

        Datos:
        {resumen}
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )

        reporte_texto = completion.choices[0].message.content.strip()

        # === 3. Crear documento Word ===
        doc = Document()
        doc.add_heading("Reporte de Análisis - InertiaX", level=1)
        doc.add_paragraph(reporte_texto)

        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        # === 4. Enviar archivo generado ===
        return send_file(
            output,
            as_attachment=True,
            download_name="Reporte_InertiaX.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        print("❌ Error en /analizar:", e)
        return jsonify({"error": str(e)}), 500
