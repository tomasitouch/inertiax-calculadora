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
        print("✅ Recibida solicitud POST /analizar")

        archivo = request.files["archivo"]
        print(f"📁 Archivo recibido: {archivo.filename}")

        df = pd.read_csv(archivo, sep=None, engine="python")
        print("📊 CSV cargado correctamente")
        print(df.head())

        resumen = df.describe().to_string()

        prompt = f"""
        Eres un analista deportivo de InertiaX. Analiza los datos de entrenamiento y
        genera un resumen técnico profesional, destacando velocidad, potencia y
        recomendaciones.
        
        Datos:
        {resumen}
        """

        print("🧠 Enviando datos al modelo...")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )

        reporte_texto = completion.choices[0].message.content.strip()
        print("📄 Respuesta IA recibida correctamente")

        from docx import Document
        import io
        doc = Document()
        doc.add_heading("Reporte de Análisis - InertiaX", level=1)
        doc.add_paragraph(reporte_texto)

        output = io.BytesIO()
        doc.save(output)
        output.seek(0)

        print("💾 Documento Word generado correctamente")
        return send_file(
            output,
            as_attachment=True,
            download_name="Reporte_InertiaX.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

    except Exception as e:
        print("❌ Error en /analizar:", str(e))
        return jsonify({"error": str(e)}), 500




