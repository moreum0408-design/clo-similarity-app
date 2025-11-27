import os
import re
import math
from collections import defaultdict
from textwrap import shorten

from flask import Flask, request, render_template_string
import pandas as pd
import numpy as pd
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

REQUIRED = ["COLLEGE", "Program", "COURSE", "CLO_TEXT"]
for col in REQUIRED:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
df["COURSE"] = df["COURSE"].astype(str)
df["Program"] = df["Program"].astype(str)

# build dropdown
COURSES = sorted(df["COURSE"].unique())

def build_clo_label(row, max_len=90):
    base = f"{row['COURSE']} – {row['CLO_TEXT']}"
    return shorten(base, width=max_len, placeholder="…")

CLO_OPTIONS = [
    {"id": idx, "label": build_clo_label(row)}
    for idx, row in df.iterrows()
]

# ---------------------------------------------------------
# TF-IDF (scikit-learn)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["CLO_TEXT"])   # shape (N, features)

# ---------------------------------------------------------
# Similarity Functions
# ---------------------------------------------------------
def cosine_sim_rows(i, j):
    v1 = tfidf_matrix[i]
    v2 = tfidf_matrix[j]
    return float(cosine_similarity(v1, v2)[0, 0])

COURSE_TO_ROWS = defaultdict(list)
for idx, row in df.iterrows():
    COURSE_TO_ROWS[row["COURSE"]].append(idx)

def mean_vector(indices):
    if not indices:
        return None
    sub = tfidf_matrix[indices]
    mean = sub.mean(axis=0)
    return mean

def course_vs_course_similarity(course_a, course_b):
    idx_a = COURSE_TO_ROWS.get(course_a, [])
    idx_b = COURSE_TO_ROWS.get(course_b, [])

    if not idx_a or not idx_b:
        return None

    mean_a = mean_vector(idx_a)
    mean_b = mean_vector(idx_b)

    # convert possible np.matrix to regular ndarrays, then keep them 2D
    mean_a = np.asarray(mean_a).reshape(1, -1)
    mean_b = np.asarray(mean_b).reshape(1, -1)

    overall = float(cosine_similarity(mean_a, mean_b)[0, 0])

    details = []
    for i_local, i_global in enumerate(idx_a):
        best_j_local = sim_matrix[i_local].argmax()
        best_j_global = idx_b[best_j_local]
        best_sim = sim_matrix[i_local][best_j_local]

        details.append(
            {
                "clo_a": df.loc[i_global, "CLO_TEXT"],
                "clo_b": df.loc[best_j_global, "CLO_TEXT"],
                "similarity": float(best_sim),
            }
        )

    return {"overall": overall, "details": details}

def parse_course_prefix_and_level(course_code):
    if not course_code:
        return None, None
    code = course_code.replace(" ", "")
    m_pre = re.match(r"^([A-Za-z]+)", code)
    prefix = m_pre.group(1) if m_pre else None
    m_num = re.search(r"(\d+)", code)
    if not m_num:
        return prefix, None
    num = m_num.group(1)
    level = int(num[0]) * 100
    return prefix, level

def course_vs_level_similarity(course_a):
    prefix, level = parse_course_prefix_and_level(course_a)
    if prefix is None or level is None:
        return []

    candidates = []
    for c in COURSES:
        if c == course_a:
            continue
        p2, l2 = parse_course_prefix_and_level(c)
        if p2 == prefix and l2 == level:
            sim = course_vs_course_similarity(course_a, c)
            if sim is not None:
                candidates.append(
                    {
                        "course_b": c,
                        "overall": sim["overall"],
                    }
                )

    candidates.sort(key=lambda x: x["overall"], reverse=True)
    return candidates

# ---------------------------------------------------------
# FLASK
# ---------------------------------------------------------
app = Flask(__name__)

HTML = """<!doctype html>
<html>
<head>
<title>CLO Similarity Tool</title>
<style>
body { font-family: Arial; margin: 20px; }
.result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
table { border-collapse: collapse; width: 100%; margin-top: 10px;}
th, td { border: 1px solid #ccc; padding: 8px; }
.sim { font-weight:bold; }
</style>
</head>
<body>
<h1>CLO Similarity Tool</h1>

<form method="post">
    <p><strong>Choose mode:</strong></p>
    <label><input type="radio" name="mode" value="course" {% if mode=='course' %}checked{% endif %}> Course vs Course</label><br>
    <label><input type="radio" name="mode" value="course_level" {% if mode=='course_level' %}checked{% endif %}> Course vs Same-Level</label><br>
    <label><input type="radio" name="mode" value="clo" {% if mode=='clo' %}checked{% endif %}> CLO vs CLO</label>

    {% if mode != 'clo' %}
        <p>Course A:</p>
        <select name="course_a">
            <option value="">-- select --</option>
            {% for c in courses %}
                <option value="{{c}}" {% if course_a==c %}selected{% endif %}>{{c}}</option>
            {% endfor %}
        </select>

        {% if mode=='course' %}
            <p>Course B:</p>
            <select name="course_b">
                <option value="">-- select --</option>
                {% for c in courses %}
                    <option value="{{c}}" {% if course_b==c %}selected{% endif %}>{{c}}</option>
                {% endfor %}
            </select>
        {% endif %}
    {% else %}
        <p>CLO A:</p>
        <select name="clo_a">
            <option value="">-- select --</option>
            {% for clo in clos %}
                <option value="{{clo.id}}" {% if clo_a==clo.id|string %}selected{% endif %}>{{clo.label}}</option>
            {% endfor %}
        </select>

        <p>CLO B:</p>
        <select name="clo_b">
            <option value="">-- select --</option>
            {% for clo in clos %}
                <option value="{{clo.id}}" {% if clo_b==clo.id|string %}selected{% endif %}>{{clo.label}}</option>
            {% endfor %}
        </select>
    {% endif %}

    <p><input type="submit" value="Compare"></p>
</form>

{% if error %}
<div class="result"><strong>Error:</strong> {{error}}</div>
{% endif %}

{% if course_result %}
<div class="result">
    <h2>Course vs Course</h2>
    <p>Overall similarity: <span class="sim">{{course_result.overall|round(3)}}</span></p>
    <table>
        <tr><th>CLO A</th><th>CLO B</th><th>Similarity</th></tr>
        {% for row in course_result.details %}
        <tr>
            <td>{{row.clo_a}}</td>
            <td>{{row.clo_b}}</td>
            <td class="sim">{{row.similarity|round(3)}}</td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

{% if level_results %}
<div class="result">
    <h2>Course vs Same Level</hh2>
    <table>
        <tr><th>Course</th><th>Similarity</th></tr>
        {% for r in level_results %}
        <tr>
            <td>{{r.course_b}}</td>
            <td class="sim">{{r.overall|round(3)}}</td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

{% if clo_sim %}
<div class="result">
    <h2>CLO vs CLO</h2>
    <p><strong>{{clo_a_text}}</strong></p>
    <p><strong>{{clo_b_text}}</strong></p>
    <p>Similarity: <span class="sim">{{clo_sim|round(3)}}</span></p>
</div>
{% endif %}

</body>
</html>
"""

def sim_class(v):
    if v >= 0.75: return "high"
    if v >= 0.5: return "medium"
    return "low"

@app.route("/", methods=["GET", "POST"])
def index():
    mode = request.form.get("mode", "course")
    course_a = request.form.get("course_a", "")
    course_b = request.form.get("course_b", "")
    clo_a = request.form.get("clo_a", "")
    clo_b = request.form.get("clo_b", "")

    error = None
    course_result = None
    level_results = None
    clo_sim = None
    clo_a_text = ""
    clo_b_text = ""

    if request.method == "POST":
        if mode == "course":
            if not course_a or not course_b:
                error = "Select both courses."
            elif course_a == course_b:
                error = "Choose two different courses."
            else:
                course_result = course_vs_course_similarity(course_a, course_b)

        elif mode == "course_level":
            if not course_a:
                error = "Select a course."
            else:
                level_results = course_vs_level_similarity(course_a)

        elif mode == "clo":
            if not clo_a or not clo_b:
                error = "Select both CLOs."
            elif clo_a == clo_b:
                error = "Choose two different CLOs."
            else:
                idx_a = int(clo_a)
                idx_b = int(clo_b)
                clo_a_text = df.loc[idx_a, "CLO_TEXT"]
                clo_b_text = df.loc[idx_b, "CLO_TEXT"]
                clo_sim = cosine_sim_rows(idx_a, idx_b)

    return render_template_string(
        HTML,
        mode=mode,
        courses=COURSES,
        clos=CLO_OPTIONS,
        course_a=course_a,
        course_b=course_b,
        clo_a=clo_a,
        clo_b=clo_b,
        course_result=course_result,
        level_results=level_results,
        clo_sim=clo_sim,
        clo_a_text=clo_a_text,
        clo_b_text=clo_b_text,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=False)
