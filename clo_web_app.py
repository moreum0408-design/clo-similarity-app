import os
import pandas as pd
from flask import Flask, render_template, request, abort
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Absolute paths
CLO_SIM_FILE = os.path.join(BASE_DIR, "clo_similarity_top.csv")          # huge
COURSE_SIM_FILE = os.path.join(BASE_DIR, "course_similarity_top.csv")    # smaller
CLO_DRILL_FILE = os.path.join(BASE_DIR, "clo_similarity_drilldown.csv")  # optional

DF_COURSE = None


def _clean(v):
    v = (v or "").strip()
    return v if v else None


def load_course_df():
    """Load the SMALL course similarity CSV into memory once."""
    global DF_COURSE
    if DF_COURSE is not None:
        return

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

    DF_COURSE.drop_duplicates(subset=["base_course", "other_course"], keep="first", inplace=True)


def _iter_clo_chunks(csv_path: str, chunksize: int = 200_000):
    """
    Stream big CSV in chunks and only load needed columns.
    Prevents OOM on Render.
    """
    usecols = ["base_course", "base_clo", "compare_course", "compare_clo", "similarity"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["base_course"] = chunk["base_course"].astype(str).str.strip()
        chunk["compare_course"] = chunk["compare_course"].astype(str).str.strip()
        chunk["base_clo"] = chunk["base_clo"].astype(str).fillna("").str.strip()
        chunk["compare_clo"] = chunk["compare_clo"].astype(str).fillna("").str.strip()
        chunk["similarity"] = pd.to_numeric(chunk["similarity"], errors="coerce").fillna(0.0)
        yield chunk


@lru_cache(maxsize=512)
def get_clo_options_for_course(base_course: str):
    """Return base CLO dropdown options for a selected base_course (chunk scan + cache)."""
    if not os.path.exists(CLO_SIM_FILE):
        raise FileNotFoundError(
            f"Missing {os.path.basename(CLO_SIM_FILE)}. Run precompute_similarity.py first."
        )

    seen = set()
    out = []

    for chunk in _iter_clo_chunks(CLO_SIM_FILE):
        sub = chunk.loc[chunk["base_course"] == base_course, "base_clo"]
        for clo in sub.tolist():
            if clo and clo not in seen:
                seen.add(clo)
                out.append(clo)

    return out


@lru_cache(maxsize=2048)
def get_top_clo_matches(base_course: str, base_clo: str, topn: int = 20):
    """Top N CLO matches for one base CLO (chunk scan + cache)."""
    if not os.path.exists(CLO_SIM_FILE):
        raise FileNotFoundError(
            f"Missing {os.path.basename(CLO_SIM_FILE)}. Run precompute_similarity.py first."
        )

    hits = []
    for chunk in _iter_clo_chunks(CLO_SIM_FILE):
        m = (chunk["base_course"] == base_course) & (chunk["base_clo"] == base_clo)
        if m.any():
            hits.append(chunk.loc[m, ["compare_course", "compare_clo", "similarity"]])

    if not hits:
        return []

    df = pd.concat(hits, ignore_index=True)
    df.drop_duplicates(subset=["compare_course", "compare_clo"], keep="first", inplace=True)
    df.sort_values("similarity", ascending=False, inplace=True)
    df = df.head(topn)

    return df.to_dict(orient="records")


@lru_cache(maxsize=4096)
def get_course_detail_rows(base_course: str, base_clo: str, compare_course: str, limit: int = 200):
    """
    Drilldown: show CLOs within the clicked compare_course.
    Prefer clo_similarity_drilldown.csv if it exists; otherwise fallback to clo_similarity_top.csv.
    """
    csv_path = CLO_DRILL_FILE if os.path.exists(CLO_DRILL_FILE) else CLO_SIM_FILE
    using_drill = (csv_path == CLO_DRILL_FILE)

    hits = []
    for chunk in _iter_clo_chunks(csv_path):
        m = (
            (chunk["base_course"] == base_course)
            & (chunk["base_clo"] == base_clo)
            & (chunk["compare_course"] == compare_course)
        )
        if m.any():
            hits.append(chunk.loc[m, ["compare_course", "compare_clo", "similarity"]])

    if not hits:
        return [], using_drill

    df = pd.concat(hits, ignore_index=True)
    df.drop_duplicates(subset=["compare_course", "compare_clo"], keep="first", inplace=True)
    df.sort_values("similarity", ascending=False, inplace=True)
    df = df.head(limit)

    return df.to_dict(orient="records"), using_drill


def get_overall_course_similarity(base_course: str, other_course: str):
    """Lookup overall course similarity from course_similarity_top.csv (already in memory)."""
    load_course_df()
    m = (DF_COURSE["base_course"] == base_course) & (DF_COURSE["other_course"] == other_course)
    row = DF_COURSE.loc[m]
    if row.empty:
        return None
    return float(row["overall"].iloc[0])


@app.route("/", methods=["GET", "POST"])
def index():
    load_course_df()

    courses = sorted(DF_COURSE["base_course"].unique().tolist())

    selected_course = _clean(request.args.get("course"))
    selected_clo = _clean(request.args.get("clo"))

    if request.method == "POST":
        selected_course = _clean(request.form.get("course"))
        selected_clo = _clean(request.form.get("clo"))

    clo_options = []
    course_results = []
    clo_results = []

    if selected_course:
        clo_options = get_clo_options_for_course(selected_course)

        course_results = (
            DF_COURSE.loc[DF_COURSE["base_course"] == selected_course]
            .sort_values("overall", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )

    if selected_course and selected_clo:
        clo_results = get_top_clo_matches(selected_course, selected_clo, topn=20)

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
    load_course_df()

    base_course = _clean(request.args.get("base_course"))
    base_clo = _clean(request.args.get("base_clo"))
    if not base_course or not base_clo:
        abort(400, "Missing base_course or base_clo")

    course_code = course_code.strip()

    # overall similarity for the clicked course
    overall = get_overall_course_similarity(base_course, course_code)

    # limit for drilldown rows
    limit_raw = _clean(request.args.get("limit"))
    try:
        limit = int(limit_raw) if limit_raw else 200
    except Exception:
        limit = 200
    limit = max(1, min(500, limit))

    rows, using_drill = get_course_detail_rows(base_course, base_clo, course_code, limit=limit)

    return render_template(
        "course_detail.html",
        course=course_code,
        base_course=base_course,
        base_clo=base_clo,
        overall=overall,
        rows=rows,
        drilldown_enabled=using_drill,
        limit=limit,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
