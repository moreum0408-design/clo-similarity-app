import os
import pandas as pd
from flask import Flask, render_template, request, abort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Files (absolute paths so Render working-dir never matters)
CLO_SIM_FILE = os.path.join(BASE_DIR, "clo_similarity_top.csv")
COURSE_SIM_FILE = os.path.join(BASE_DIR, "course_similarity_top.csv")
CLO_DRILL_FILE = os.path.join(BASE_DIR, "clo_similarity_drilldown.csv")

DF_CLO = None
DF_COURSE = None
DF_DRILL = None


def _normalize_clo_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_course"] = df["base_course"].astype(str).str.strip()
    df["compare_course"] = df["compare_course"].astype(str).str.strip()
    df["base_clo"] = df["base_clo"].astype(str).fillna("").str.strip()
    df["compare_clo"] = df["compare_clo"].astype(str).fillna("").str.strip()
    df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce").fillna(0.0)
    return df


def load_data():
    global DF_CLO, DF_COURSE, DF_DRILL

    # CLO top list used on main page
    if DF_CLO is None:
        if not os.path.exists(CLO_SIM_FILE):
            raise FileNotFoundError(
                f"Missing {os.path.basename(CLO_SIM_FILE)}. Run precompute_similarity.py first."
            )

        DF_CLO = pd.read_csv(CLO_SIM_FILE)
        required = {"base_course", "base_clo", "compare_course", "compare_clo", "similarity"}
        missing = required - set(DF_CLO.columns)
        if missing:
            raise ValueError(f"{os.path.basename(CLO_SIM_FILE)} missing columns: {sorted(missing)}")

        DF_CLO = _normalize_clo_df(DF_CLO)
        DF_CLO.drop_duplicates(
            subset=["base_course", "base_clo", "compare_course", "compare_clo"],
            keep="first",
            inplace=True,
        )

    # Course top list used on main page
    if DF_COURSE is None:
        if not os.path.exists(COURSE_SIM_FILE):
            raise FileNotFoundError(
                f"Missing {os.path.basename(COURSE_SIM_FILE)}. Run precompute_similarity.py first."
            )

        DF_COURSE = pd.read_csv(COURSE_SIM_FILE)
        required = {"base_course", "other_course", "overall"}
        missing = required - set(DF_COURSE.columns)
        if missing:
            raise ValueError(f"{os.path.basename(COURSE_SIM_FILE)} missing columns: {sorted(missing)}")

        DF_COURSE["base_course"] = DF_COURSE["base_course"].astype(str).str.strip()
        DF_COURSE["other_course"] = DF_COURSE["other_course"].astype(str).str.strip()
        DF_COURSE["overall"] = pd.to_numeric(DF_COURSE["overall"], errors="coerce").fillna(0.0)

        DF_COURSE.drop_duplicates(
            subset=["base_course", "other_course"],
            keep="first",
            inplace=True,
        )

    # Drilldown CLO list used on /course/<course_code>
    if DF_DRILL is None:
        if os.path.exists(CLO_DRILL_FILE):
            DF_DRILL = pd.read_csv(CLO_DRILL_FILE)
            required = {"base_course", "base_clo", "compare_course", "compare_clo", "similarity"}
            missing = required - set(DF_DRILL.columns)
            if missing:
                raise ValueError(
                    f"{os.path.basename(CLO_DRILL_FILE)} missing columns: {sorted(missing)}"
                )

            DF_DRILL = _normalize_clo_df(DF_DRILL)
            DF_DRILL.drop_duplicates(
                subset=["base_course", "base_clo", "compare_course", "compare_clo"],
                keep="first",
                inplace=True,
            )
        else:
            DF_DRILL = None  # fallback mode


def _clean(v: str | None) -> str | None:
    v = (v or "").strip()
    return v if v else None


@app.route("/", methods=["GET", "POST"])
def index():
    load_data()

    courses = sorted(DF_CLO["base_course"].unique().tolist())

    # STATE RESTORE: allow GET params to repopulate selections
    selected_course = _clean(request.args.get("course"))
    selected_clo = _clean(request.args.get("clo"))

    # POST overrides GET when user submits form
    if request.method == "POST":
        selected_course = _clean(request.form.get("course"))
        selected_clo = _clean(request.form.get("clo"))

    clo_options = []
    course_results = []
    clo_results = []

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

    base_course = _clean(request.args.get("base_course"))
    base_clo = _clean(request.args.get("base_clo"))
    if not base_course or not base_clo:
        abort(400, "Missing base_course or base_clo")

    course_code = course_code.strip()

    # How many CLO rows to show on drilldown
    limit_raw = (request.args.get("limit") or "").strip()
    try:
        limit = int(limit_raw) if limit_raw else 200
    except Exception:
        limit = 200
    limit = max(1, min(500, limit))

    # Prefer drilldown DF, fallback to top DF
    src = DF_DRILL if DF_DRILL is not None else DF_CLO

    rows = (
        src.loc[
            (src["base_course"] == base_course)
            & (src["base_clo"] == base_clo)
            & (src["compare_course"] == course_code)
        ]
        .drop_duplicates(subset=["compare_course", "compare_clo"])
        .sort_values("similarity", ascending=False)
        .head(limit)
        .to_dict(orient="records")
    )

    drilldown_enabled = DF_DRILL is not None

    return render_template(
        "course_detail.html",
        course=course_code,
        base_course=base_course,
        base_clo=base_clo,
        rows=rows,
        drilldown_enabled=drilldown_enabled,
        limit=limit,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
