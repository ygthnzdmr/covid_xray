import os
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
import tensorflow as tf

IMG_SIZE = (64, 64)
THRESHOLD = 0.5

model = tf.keras.models.load_model("covid_ann_model.keras")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    filename = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if not file or file.filename == "":
            error = "Lütfen bir görüntü dosyası seçin."
        else:
            try:
                filename = file.filename
                img = Image.open(file.stream).convert("L")
                img = img.resize(IMG_SIZE)
                x = np.array(img) / 255.0
                x = x.reshape(1, 64, 64, 1)

                prob = float(model.predict(x, verbose=0)[0][0])
                probability = f"{prob * 100:.2f}"
                result = "COVID" if prob >= THRESHOLD else "NORMAL"
            except Exception as e:
                error = f"Görüntü işlenirken hata oluştu: {e}"

    return render_template("index.html", result=result, probability=probability,
                           filename=filename, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)