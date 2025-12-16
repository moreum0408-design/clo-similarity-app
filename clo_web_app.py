import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ======================
# LOAD DATA
# ======================
ALL_CLOS_FILE = "All CLOs copy.xlsx"
CLO_SIM_FILE = "clo_similarity_top.csv"
COURSE_SIM_FILE = "course_similarity_top.csv"

df_all = pd.read_excel(ALL_CLOS_FILE)
df_clo_sim = pd.read_csv(CLO_SIM_FILE)
df_course_sim = pd.read_csv(COURSE_SIM_FILE)

df_all["COURSE"] = df_all["COURSE"].astype(str)
df_all["CLO_TEXT"] = df_all["CLO_TEXT"].astype(str)

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def index():
    courses = sorted(df_all["COURSE"].unique())
    selected_course = None
    selected_clo = None
    results = []

    if request.method == "POST":
        selected_course = request.form["course"]
        selected_clo = request.form["clo"]

        # base CLO index
        base_idx = df_all[
            (df_all["COURSE"] == selected_course) &
            (df_all["CLO_TEXT"] == selected_clo)
        ].index[0]

        # ======================
        # 🔥 THIS IS THE FIX 🔥
        # NO GROUPING
        # PURE TOP 20 BY SIMILARITY
        # ======================
        results = (
            df_clo_sim[df_clo_sim["base_idx"] == base_idx]
            .merge(
                df_all[["COURSE", "CLO_TEXT"]],
                left_on="other_idx",
                right_index=True
            )
            .sort_values("similarity", ascending=False)
            .head(20)
        )

    return render_template(
        "index.html",
        courses=courses,
        selected_course=selected_course,
        selected_clo=selected_clo,
        results=results
    )


@app.route("/get_clos")
def get_clos():
    course = request.args.get("course")
    clos = df_all[df_all["COURSE"] == course]["CLO_TEXT"].tolist()
    return {"clos": clos}


if __name__ == "__main__":
    app.run(debug=False)
