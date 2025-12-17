from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

app = Flask(__name__, template_folder="templates")

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"
CLO_SIM_FILE = "clo_similarity_top.csv"
COURSE_SIM_FILE = "course_similarity_top.csv"

VECTORIZER = HashingVectorizer(
    n_features=4096,
    alternate_sign=False,
    norm=None,
    stop_words="english",
)

_df_clo = None
_df_course = None
_course_to_clos = None
_course_col = None


def _load_top_csvs():
    global _df_clo, _df_course
    if _df_clo is None:
        if not os.path.exists(CLO_SIM_FILE):
            raise FileNotFoundError(f"Missing {CLO_SIM_FILE} in deployed filesystem.")
        _df_clo = pd.read_csv(CLO_SIM_FILE)

        required = {"base_course", "base_clo", "compare_course", "compare_clo", "similarity"}
        missing = required - set(_df_clo.columns)
        if missing:
            raise ValueError(f"{CLO_SIM_FILE} missing columns: {sorted(missing)}")

    if _df_course is None:
        if not os.path.exists(COURSE_SIM_FILE):
            raise FileNotFoundError(f"Missing {COURSE_SIM_FILE} in deployed filesystem.")
        _df_course = pd.read_csv(COURSE_SIM_FILE)

        required = {"base_course", "other_course", "overall"}
        missing = required - set(_df_course.columns)
        if missing:
            raise ValueError(f"{COURSE_SIM_FILE} missing columns: {sorted(missing)}")


def _load_all_clos():
    global _course_to_clos, _course_col
    if _course_to_clos is not None:
        return

    if not os.path.exists(DATA_FILE):
        # On Render, you may NOT have the Excel file if you .gitignore’d it.
        # We only need this for the "click course -> show all CLOs" page.
        _course_to_clos = {}
        _course_col = "COURSE"
        return

    df_all = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    _course_col = "COURSE" if "COURSE" in df_all.columns else "Course"
    if "CLO_TEXT" not in df_all.columns:
        raise ValueError("Excel is missing CLO_TEXT column.")

    df_all[_course_col] = df_all[_course_col].fillna("").astype(str).str.strip()
    df_all["CLO_TEXT"] = df_all["CLO_TEXT"].fillna("").astype(str).str.strip()

    # HARD DEDUPE: (course, clo_text)
    df_all = df_all.drop_duplicates(subset=[_course_col, "CLO_TEXT"]).reset_index(drop=True)

    _course_to_clos = (
        df_all.groupby(_course_col)["CLO_TEXT"]
        .apply(list)
        .to_dict()
    )


def _cosine_sim(base_text: str, compare_texts: list[str]) -> np.ndarray:
    base_vec = VECTORIZER.transform([base_text])
    base_vec = normalize(base_vec)

    mat = VECTORIZER.transform(compare_texts)
    mat = normalize(mat)

    return (base_vec @ mat.T).toarray().ravel()


@app.route("/", methods=["GET", "POST"])
def index():
    try:
        _load_top_csvs()
        courses = sorted(_df_clo["base_course"].unique())

        selected_course = None
        selected_clo = None
        clo_options = []
        clo_results = []
        course_results = []

        if request.method == "POST":
            selected_course = request.form.get("course") or None
            selected_clo = request.form.get("clo") or None

            if selected_course:
                clo_options = (
                    _df_clo[_df_clo["base_course"] == selected_course]["base_clo"]
                    .drop_duplicates()
                    .tolist()
                )

                course_results = (
                    _df_course[_df_course["base_course"] == selected_course]
                    .sort_values("overall", ascending=False)
                    .head(20)
                    .to_dict(orient="records")
                )

            if selected_course and selected_clo:
                clo_results = (
                    _df_clo[
                        (_df_clo["base_course"] == selected_course)
                        & (_df_clo["base_clo"] == selected_clo)
                    ]
                    .drop_duplicates(subset=["compare_course", "compare_clo"])
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

    except Exception as e:
        # This prints the real cause in Render logs
        app.logger.exception("Crash on /")
        return f"<h1>Internal Server Error</h1><pre>{e}</pre>", 500


@app.route("/course/<course_code>")
def course_detail(course_code):
    try:
        _load_top_csvs()
        _load_all_clos()

        base_course = request.args.get("base_course") or ""
        base_clo = request.args.get("base_clo") or ""

        clos = _course_to_clos.get(course_code, [])

        rows = []
        if base_clo and clos:
            sims = _cosine_sim(base_clo, clos)
            # guaranteed unique CLO_TEXT already, but keep a safety dedupe:
            seen = set()
            for clo, sim in sorted(zip(clos, sims), key=lambda x: x[1], reverse=True):
                if clo in seen:
                    continue
                seen.add(clo)
                rows.append({"compare_clo": clo, "similarity": float(sim)})

        return render_template(
            "course_detail.html",
            course=course_code,
            base_course=base_course,
            base_clo=base_clo,
            rows=rows,
        )

    except Exception as e:
        app.logger.exception("Crash on /course/<course_code>")
        return f"<h1>Internal Server Error</h1><pre>{e}</pre>", 500
