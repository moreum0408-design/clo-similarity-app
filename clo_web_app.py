import os
import re
from collections import defaultdict
from textwrap import shorten

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Put it in the same folder as clo_web_app.py."
    )

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

required_cols = ["COLLEGE", "Program", "CLO_TEXT"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# handle Course / COURSE flexibly
if "COURSE" in df.columns:
    COURSE_COL = "COURSE"
elif "Course" in df.columns:
    COURSE_COL = "Course"
else:
    raise ValueError("Could not find a COURSE or Course column in the Excel file.")

df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
df[COURSE_COL] = df[COURSE_COL].astype(str)
df["Program"] = df["Program"].astype(str)

COURSES = sorted(df[COURSE_COL].unique())

# ---------------------------------------------------------
# TF-IDF (computed ONCE at startup)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["CLO_TEXT"])  # sparse (N, D)

# ---------------------------------------------------------
# Helper structures
# ---------------------------------------------------------
COURSE_TO_ROWS = defaultdict(list)
for idx, row in df.iterrows():
    COURSE_TO_ROWS[row[COURSE_COL]].append(int(idx))


def parse_course_level(course_code: str):
    """
    From e.g. 'CJUS 887' -> 800
    Only uses the first digit of the numeric part (hundreds level),
    ignores department prefix entirely.
    """
    if not course_code:
        return None
    code = course_code.replace(" ", "")
    m = re.search(r"(\d+)", code)
    if not m:
        return None
    num = m.group(1)
    if not num or not num[0].isdigit():
        return None
    return int(num[0]) * 100


def mean_vector(indices):
    """1 x D ndarray mean TF-IDF vector for given row indices."""
    if not indices:
        return None
    sub = tfidf_matrix[indices]
    arr = np.asarray(sub.mean(axis=0)).ravel()
    return arr.reshape(1, -1)


def course_vs_level_similarity(course_a):
    """
    Overall similarity: one course vs ALL same-level courses (e.g. any 4xx or 8xx),
    regardless of department prefix.
    """
    base_level = parse_course_level(course_a)
    if base_level is None:
        return []

    idx_a = COURSE_TO_ROWS.get(course_a, [])
    if not idx_a:
        return []

    mean_a = mean_vector(idx_a)
    if mean_a is None:
        return []

    results = []
    for c in COURSES:
        if c == course_a:
            continue
        if parse_course_level(c) != base_level:
            continue
        idx_b = COURSE_TO_ROWS.get(c, [])
        if not idx_b:
            continue
        mean_b = mean_vector(idx_b)
        if mean_b is None:
            continue
        overall = float(cosine_similarity(mean_a, mean_b)[0, 0])
        results.append({"course_b": c, "overall": overall})

    results.sort(key=lambda x: x["overall"], reverse=True)
    return results


def clo_vs_same_level_clos(base_idx, max_results=100):
    """
    ONE CLO index vs ALL CLOs of all OTHER same-level courses (e.g. any 4xx/8xx),
    excluding:
      - CLOs from the same course
      - duplicate (course, CLO_TEXT) rows (keep the highest similarity)
    """
    base_course = df.loc[base_idx, COURSE_COL]
    base_level = parse_course_level(base_course)
    if base_level is None:
        return []

    base_vec = tfidf_matrix[base_idx]  # 1 x D

    comp_indices = []
    for idx, course in df[COURSE_COL].items():
        # skip all CLOs from the same course
        if course == base_course:
            continue
        # only same level across ALL departments
        if parse_course_level(course) != base_level:
            continue
        comp_indices.append(idx)

    if not comp_indices:
        return []

    comp_mat = tfidf_matrix[comp_indices]             # M x D
    sims = cosine_similarity(base_vec, comp_mat)[0]   # length M

    # First build raw results
    raw_rows = []
    for idx, sim in zip(comp_indices, sims):
        raw_rows.append(
            {
                "course": df.loc[idx, COURSE_COL],
                "clo_text": df.loc[idx, "CLO_TEXT"],
                "similarity": float(sim),
            }
        )

    # Deduplicate: keep only the highest similarity per (course, clo_text)
    best_by_key = {}
    for r in raw_rows:
        key = (r["course"], r["clo_text"])
        if key not in best_by_key or r["similarity"] > best_by_key[key]["similarity"]:
            best_by_key[key] = r

    rows = sorted(best_by_key.values(), key=lambda x: x["similarity"], reverse=True)
    return rows[:max_results]


def course_clo_options(course):
    """
    Return list of {id, label} CLO options for a given course,
    for the second dropdown.
    """
    if not course:
        return []
    subset = df[df[COURSE_COL] == course]
    options = []
    for idx, row in subset.iterrows():
        label = shorten(row["CLO_TEXT"], width=120, placeholder="…")
        options.append({"id": int(idx), "label": label})
    return options


# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)

HTML = """<!doctype html>
<html>
<head>
<title>CLO Similarity Tool</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
.result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
table { border-collapse: collapse; width: 100%; margin-top: 10px;}
th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
.sim { font-weight:bold; }
</style>
</head>
<body>
<h1>CLO Similarity Tool</h1>

<form method="post">
    <p><strong>Select a course and one CLO from that course.</strong></p>

    <p>Course:</p>
    <select name="course_a" onchange="this.form.submit()">
        <option value="">-- select course --</option>
        {% for c in courses %}
            <option value="{{c}}" {% if course_a==c %}selected{% endif %}>{{c}}</option>
        {% endfor %}
    </select>

    <p>CLO (from selected course):</p>
    <select name="clo_idx">
        <option value="">{% if not course_a %}-- select course first --{% else %}-- select CLO --{% endif %}</option>
        {% for clo in course_clos %}
            <option value="{{clo.id}}" {% if clo_idx==clo.id|string %}selected{% endif %}>{{clo.label}}</option>
        {% endfor %}
    </select>

    <p><input type="submit" value="Compare"></p>
</form>

{% if error %}
<div class="result"><strong>Error:</strong> {{error}}</div>
{% endif %}

{% if base_clo_text %}
<div class="result">
    <h2>Selected CLO</h2>
    <p><strong>Course:</strong> {{ course_a }}</p>
    <p><strong>CLO:</strong> {{ base_clo_text }}</p>
</div>
{% endif %}

{% if course_level_results %}
<div class="result">
    <h2>Overall Course vs Same-Level Courses (all departments)</h2>
    <table>
        <tr><th>Course</th><th>Overall Similarity</th></tr>
        {% for r in course_level_results %}
        <tr>
            <td>{{ r.course_b }}</td>
            <td class="sim">{{ r.overall|round(3) }}</td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

{% if clo_results %}
<div class="result">
    <h2>This CLO vs All CLOs in Same-Level Courses (other courses only)</h2>
    <table>
        <tr><th>Course</th><th>CLO</th><th>Similarity</th></tr>
        {% for r in clo_results %}
        <tr>
            <td>{{ r.course }}</td>
            <td>{{ r.clo_text }}</td>
            <td class="sim">{{ r.similarity|round(3) }}</td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    course_a = request.form.get("course_a", "")
    clo_idx = request.form.get("clo_idx", "")

    error = None
    base_clo_text = ""
    course_level_results = []
    clo_results = []

    # CLO dropdown options depend on selected course
    course_clos = course_clo_options(course_a)

    if request.method == "POST":
        if not course_a:
            error = "Select a course."
        elif not clo_idx:
            # User just changed course; we only needed to repopulate CLO dropdown.
            # No error and no similarity calculations yet.
            pass
        else:
            base_idx = int(clo_idx)
            # sanity check: ensure this CLO actually belongs to the selected course
            if df.loc[base_idx, COURSE_COL] != course_a:
                error = "Selected CLO does not belong to the chosen course."
            else:
                base_clo_text = df.loc[base_idx, "CLO_TEXT"]
                course_level_results = course_vs_level_similarity(course_a)
                clo_results = clo_vs_same_level_clos(base_idx)

    return render_template_string(
        HTML,
        courses=COURSES,
        course_a=course_a,
        clo_idx=clo_idx,
        course_clos=course_clos,
        error=error,
        base_clo_text=base_clo_text,
        course_level_results=course_level_results,
        clo_results=clo_results,
    )


if __name__ == "__main__":
    app.run(debug=False)
