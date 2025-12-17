import os
import pandas as pd
from flask import Flask, render_template, request, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

CLO_SIM_FILE = "clo_similarity_top.csv"
COURSE_SIM_FILE = "course_similarity_top.csv"

DF_CLO = None
DF_COURSE = None


def load_data():
    global DF_CLO, DF_COURSE

    if DF_CLO is None:
        if not os.path.exists(CLO_SIM_FILE):
            raise FileNotFoundError(f"Missing {CLO_SIM_FILE}. Run precompute_similarity.py first.")
        DF_CLO = pd.read_csv(CLO_SIM_FILE)

        required = {"base_course", "base_clo", "compare_course", "compare_clo", "similarity"}
        missing = required - set(DF_CLO.columns)
        if missing:
            raise ValueError(f"{CLO_SIM_FILE} missing columns: {sorted(missing)}")

        # normalize
        DF_CLO["base_course"] = DF_CLO["base_course"].astype(str).str.strip()
        DF_CLO["compare_course"] = DF_CLO["compare_course"].astype(str).str.strip()
        DF_CLO["base_clo"] = DF_CLO["base_clo"].astype(str).fillna("").str.strip()
        DF_CLO["compare_clo"] = DF_CLO["compare_clo"].astype(str).fillna("").str.strip()
        DF_CLO["similarity"] = pd.to_numeric(DF_CLO["similarity"], errors="coerce").fillna(0.0)

        # hard dedupe (no repetition)
        DF_CLO.drop_duplicates(
            subset=["base_course", "base_clo", "compare_course", "compare_clo"],
            keep="first",
            inplace=True
        )

    if DF_COURSE is None:
        if not os.path.exists(COURSE_SIM_FILE):
            raise FileNotFoundError(f"Missing {COURSE_SIM_FILE}. Run precompute_similarity.py first.")
        DF_COURSE = pd.read_csv(COURSE_SIM_FILE)

        required = {"base_course", "other_course", "overall"}
        missing = required - set(DF_COURSE.columns)
        if missing:
            raise ValueError(f"{COURSE_SIM_FILE} missing columns: {sorted(missing)}")

        DF_COURSE["base_course"] = DF_COURSE["base_course"].astype(str).str.strip()
        DF_COURSE["other_course"] = DF_COURSE["other_course"].astype(str).str.strip()
        DF_COURSE["overall"] = pd.to_numeric(DF_COURSE["overall"], errors="coerce").fillna(0.0)

        DF_COURSE.drop_duplicates(
            subset=["base_course", "other_course"],
            keep="first",
            inplace=True
        )


@app.route("/", methods=["GET", "POST"])
def index():
    load_data()

    courses = sorted(DF_CLO["base_course"].unique().tolist())

    selected_course = None
    selected_clo = None
    clo_options = []
    course_results = []
    clo_results = []

    if request.method == "POST":
        selected_course = (request.form.get("course") or "").strip() or None
        selected_clo = (request.form.get("clo") or "").strip() or None

        if selected_course:
            clo_options = (
                DF_CLO.loc[DF_CLO["base_course"] == selected_course, "base_clo"]
                .drop_duplicates()
                .tolist()
            )

            course_results = (
                DF_COURSE.loc[DF_COURSE["base_course"] == selected_course]
                .sort_values("overall", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )

        if selected_course and selected_clo:
            clo_results = (
                DF_CLO.loc[
                    (DF_CLO["base_course"] == selected_course)
                    & (DF_CLO["base_clo"] == selected_clo)
                ]
                .sort_values("similarity", ascending=False)
                .head(20)
                .to_dict(orient="records")
            )

    return render_template(
        "index.html",
        courses=courses,
        selected_course=selected_course,
        selected_clo=selected_clo,
        clo_options=clo_options,
        course_results=course_results,
        clo_results=clo_results,
    )


@app.route("/course/<course_code>")
def course_detail(course_code):
    load_data()

    base_course = (request.args.get("base_course") or "").strip()
    base_clo = (request.args.get("base_clo") or "").strip()

    if not base_course or not base_clo:
        abort(400, "Missing base_course or base_clo")

    course_code = course_code.strip()

    # show ALL CLOs in that clicked course with similarity to the selected base CLO
    rows = (
        DF_CLO.loc[
            (DF_CLO["base_course"] == base_course)
            & (DF_CLO["base_clo"] == base_clo)
            & (DF_CLO["compare_course"] == course_code)
        ]
        .drop_duplicates(subset=["compare_course", "compare_clo"])
        .sort_values("similarity", ascending=False)
        .to_dict(orient="records")
    )

    return render_template(
        "course_detail.html",
        course=course_code,
        base_course=base_course,
        base_clo=base_clo,
        rows=rows,
    )


if __name__ == "__main__":
    # local only
    app.run(host="127.0.0.1", port=5000, debug=False)
