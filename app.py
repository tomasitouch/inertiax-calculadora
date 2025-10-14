import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = "uploads"

# 🔑 Clave OpenRouter (guárdala en variable de entorno, no en código)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analizar", methods=["POST"])
def analizar():
    if "archivo" not in request.files:
        return jsonify({"error": "No se subió ningún archivo."}), 400

    file = request.files["archivo"]
    if file.filename == "":
        return jsonify({"error": "Archivo inválido."}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Leer CSV
    df = pd.read_csv(filepath)
    resumen = df.describe().to_dict()

    # Generar reporte con IA
    prompt = f"""
    Eres un analista deportivo. Genera un reporte profesional en español
    a partir del siguiente resumen estadístico de rendimiento de un atleta:
    {resumen}
    El reporte debe incluir conclusiones, fortalezas, debilidades y sugerencias.
    """

    completion = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[{"role": "user", "content": prompt}]
    )

    reporte = completion.choices[0].message["content"]

    return jsonify({"reporte": reporte})
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
