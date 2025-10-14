import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/calcular", methods=["POST"])
def calcular():
    data = request.json
    fuerza = float(data.get("fuerza", 0))
    velocidad = float(data.get("velocidad", 0))
    potencia = fuerza * velocidad
    return jsonify({"resultado": potencia})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
