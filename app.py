from flask import Flask, render_template, request
from adaptive_modulation_sim import run_adaptive

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        snr = float(request.form["snr"])
        result = run_adaptive(snr)

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
