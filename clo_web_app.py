import os
import re
from textwrap import shorten

from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_FILE = "All CLOs copy.xlsx"      # same folder as this .py
SHEET_NAME = "All CLOs_data (1)"      # the sheet with COLLEGE / Program / Course / CLO_TEXT

# ---------------------------------------------------------
# LOAD DATA ONCE (for speed)
# ---------------------------------------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find {DATA_FILE}. Put it in the same folder as clo_web_app.py."
    )

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

# Normalize column names
df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]

REQUIRED_COLS = ["COLLEGE", "PROGRAM", "COURSE", "CLO_TEXT"]
missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    raise ValueError(
        f"Sheet '{SHEET_NAME}' must contain columns {REQUIRED_COLS}. Missing: {missing}. "
        f"Current columns: {list(df.columns)}"
    )

for c in REQUIRED_COLS:
    df[c] = df[c].astype(str).str.strip()

df["ROW_ID"] = df.index

# Unique course list for dropdowns
COURSE_LIST = sorted(df["COURSE"].unique())

# CLO dropdown options
def build_clo_label(row, max_len=90):
    base = f"{row['COURSE']} – {row['CLO_TEXT']}"
    return shorten(base, width=max_len, placeholder="…")

CLO_OPTIONS = [
    {"id": int(row["ROW_ID"]), "label": build_clo_label(row)}
    for _, row in df.iterrows()
]

# ---------------------------------------------------------
# PRECOMPUTE TF-IDF (MAJOR SPEEDUP)
# ---------------------------------------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["CLO_TEXT"])  # shape: (n_clos, n_features)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def cosine_sim_rows(i, j):
    """Cosine similarity between two CLOs by row index."""
    v1 = tfidf_matrix[i]
    v2 = tfidf_matrix[j]
    return float(cosine_similarity(v1, v2)[0, 0])


def course_vs_course_similarity(course_a, course_b):
    """Overall + per-CLO similarity between two courses."""
    idx_a = df.index[df["COURSE"] == course_a].tolist()
    idx_b = df.index[df["COURSE"] == course_b].tolist()
    if not idx_a or not idx_b:
        return None

    tfidf_a = tfidf_matrix[idx_a, :]
    tfidf_b = tfidf_matrix[idx_b, :]

    # overall similarity: mean vector of each course
    mean_a = tfidf_a.mean(axis=0)
    mean_b = tfidf_b.mean(axis=0)
    overall = float(cosine_similarity(mean_a, mean_b)[0, 0])

    # detailed mapping: each CLO in A -> best match in B
    sim_matrix = cosine_similarity(tfidf_a, tfidf_b)
    details = []
    for i_local, row_idx_a in enumerate(idx_a):
        best_j_local = sim_matrix[i_local].argmax()
        best_sim = float(sim_matrix[i_local, best_j_local])
        row_idx_b = idx_b[best_j_local]

        row_a = df.loc[row_idx_a]
        row_b = df.loc[row_idx_b]

        details.append(
            {
                "course_a": row_a["COURSE"],
                "clo_a": row_a["CLO_TEXT"],
                "course_b": row_b["COURSE"],
                "clo_b": row_b["CLO_TEXT"],
                "similarity": best_sim,
            }
        )

    return {"overall": overall, "details": details}


def parse_course_prefix_and_level(course_code):
    """
    From something like 'CJUS887' or 'CJUS 887' get:
    prefix='CJUS', level=800 (based on first digit of number).
    """
    if not course_code:
        return None, None
    m_prefix = re.match(r"^([A-Za-z]+)", course_code.replace(" ", ""))
    prefix = m_prefix.group(1) if m_prefix else None

    m_num = re.search(r"(\d+)", course_code)
    if not m_num:
        return prefix, None
    num = m_num.group(1)
    level_digit = int(num[0])   # 8xx -> 8
    level = level_digit * 100
    return prefix, level


def course_vs_level_similarity(course_a):
    """
    Compare one course vs ALL courses of same prefix + same 8xx/7xx level.
    Returns list of dicts sorted by overall similarity.
    """
    prefix, level = parse_course_prefix_and_level(course_a)
    if level is None or prefix is None:
        return []

    # find candidate courses in same level & prefix, excluding itself
    candidates = set()
    for course in COURSE_LIST:
        p, lev = parse_course_prefix_and_level(course)
        if p == prefix and lev == level and course != course_a:
            candidates.add(course)

    results = []
    for course_b in candidates:
        sim_info = course_vs_course_similarity(course_a, course_b)
        if sim_info is None:
            continue
        results.append(
            {
                "course_b": course_b,
                "overall": sim_info["overall"],
            }
        )

    # sort high to low similarity
    results.sort(key=lambda x: x["overall"], reverse=True)
    return results


def sim_class(score: float) -> str:
    if score >= 0.7:
        return "sim-high"
    if score >= 0.4:
        return "sim-medium"
    return "sim-low"


# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)

TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>CLO Similarity Tool</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { margin-bottom: 0.3em; }
        form { margin-bottom: 1.5em; padding: 1em; border: 1px solid #ccc; border-radius: 6px; }
        label { display: block; margin-top: 0.6em; }
        select, input[type=submit] { margin-top: 0.2em; padding: 0.3em; }
        .radio-group { margin-top: 0.6em; }
        .result-box { padding: 1em; border-radius: 6px; background: #f5f5f5; margin-top: 1em; }
        table { border-collapse: collapse; width: 100%; margin-top: 0.8em; }
        th, td { border: 1px solid #ccc; padding: 0.4em; vertical-align: top; }
        th { background: #eee; }
        .sim-score { font-weight: bold; }
        .sim-high { color: green; }
        .sim-medium { color: #c28a00; }
        .sim-low { color: #b00020; }
        .clo-text { white-space: pre-wrap; }
        .note { font-size: 0.9em; color: #555; }
    </style>
</head>
<body>
    <h1>CLO Similarity Tool</h1>
    <p class="note">
        Data is loaded from <strong>{{ data_file }}</strong> (sheet: <strong>{{ sheet_name }}</strong>) 
        once at startup for speed. If you edit the Excel file, stop this program and run it again
        so new courses/CLOs show up in the dropdowns.
    </p>

    <form method="post">
        <div class="radio-group">
            <label><input type="radio" name="mode" value="course" {{ 'checked' if mode == 'course' else '' }}> Course vs Course</label>
            <label><input type="radio" name="mode" value="course_level" {{ 'checked' if mode == 'course_level' else '' }}> Course vs Same-Level Courses (e.g., CJUS887 vs all 8xx CJUS)</label>
            <label><input type="radio" name="mode" value="clo" {{ 'checked' if mode == 'clo' else '' }}> CLO vs CLO</label>
        </div>

        {% if mode in ['course', 'course_level'] %}
            <label for="course_a">Course A:</label>
            <select name="course_a" id="course_a" required>
                <option value="">-- choose course --</option>
                {% for c in courses %}
                    <option value="{{ c }}" {% if c == course_a %}selected{% endif %}>{{ c }}</option>
                {% endfor %}
            </select>

            {% if mode == 'course' %}
                <label for="course_b">Course B:</label>
                <select name="course_b" id="course_b" required>
                    <option value="">-- choose course --</option>
                    {% for c in courses %}
                        <option value="{{ c }}" {% if c == course_b %}selected{% endif %}>{{ c }}</option>
                    {% endfor %}
                </select>
            {% endif %}
        {% else %}
            <label for="clo_a">CLO A:</label>
            <select name="clo_a" id="clo_a" required>
                <option value="">-- choose CLO --</option>
                {% for opt in clos %}
                    <option value="{{ opt.id }}" {% if opt.id|string == clo_a %}selected{% endif %}>
                        {{ opt.label }}
                    </option>
                {% endfor %}
            </select>

            <label for="clo_b">CLO B:</label>
            <select name="clo_b" id="clo_b" required>
                <option value="">-- choose CLO --</option>
                {% for opt in clos %}
                    <option value="{{ opt.id }}" {% if opt.id|string == clo_b %}selected{% endif %}>
                        {{ opt.label }}
                    </option>
                {% endfor %}
            </select>
        {% endif %}

        <div style="margin-top:1em;">
            <input type="submit" value="Compare">
        </div>
    </form>

    {% if error %}
        <div class="result-box" style="background:#ffe5e5; border:1px solid #ffaaaa;">
            <strong>Error:</strong> {{ error }}
        </div>
    {% endif %}

    {% if mode == 'course' and result %}
        <div class="result-box">
            <h2>Course vs Course</h2>
            <p><strong>Course A:</strong> {{ course_a }} &nbsp; | &nbsp; <strong>Course B:</strong> {{ course_b }}</p>
            <p>Overall similarity:
                <span class="sim-score {{ overall_class }}">{{ "%.3f"|format(result.overall) }}</span>
            </p>

            <h3>Best matching CLOs (A → B)</h3>
            <table>
                <tr>
                    <th>Course A</th>
                    <th>CLO A</th>
                    <th>Course B (best match)</th>
                    <th>CLO B</th>
                    <th>Similarity</th>
                </tr>
                {% for row in result.details %}
                    <tr>
                        <td>{{ row.course_a }}</td>
                        <td class="clo-text">{{ row.clo_a }}</td>
                        <td>{{ row.course_b }}</td>
                        <td class="clo-text">{{ row.clo_b }}</td>
                        <td class="sim-score {{ row.sim_class }}">{{ "%.3f"|format(row.similarity) }}</td>
                    </tr>
                {% endfor %}
            </table>
        </div>
    {% endif %}

    {% if mode == 'course_level' and level_results is not none %}
        <div class="result-box">
            <h2>Course vs Same-Level Courses</h2>
            <p>
                <strong>Course A:</strong> {{ course_a }}<br>
                Comparing against all courses with the same prefix and level (e.g., 8xx).
            </p>
            {% if level_results %}
                <table>
                    <tr>
                        <th>Course B (same level)</th>
                        <th>Overall similarity</th>
                    </tr>
                    {% for row in level_results %}
                        <tr>
                            <td>{{ row.course_b }}</td>
                            <td class="sim-score {{ row.sim_class }}">{{ "%.3f"|format(row.overall) }}</td>
                        </tr>
                    {% endfor %}
                </table>
            {% else %}
                <p>No same-level courses found to compare against.</p>
            {% endif %}
        </div>
    {% endif %}

    {% if mode == 'clo' and clo_sim is not none %}
        <div class="result-box">
            <h2>CLO vs CLO</h2>
            <p>Similarity:
                <span class="sim-score {{ clo_sim_class }}">{{ "%.3f"|format(clo_sim) }}</span>
            </p>
            <h3>CLO A</h3>
            <p class="clo-text"><strong>{{ clo_a_course }}</strong><br>{{ clo_a_text }}</p>
            <h3>CLO B</h3>
            <p class="clo-text"><strong>{{ clo_b_course }}</strong><br>{{ clo_b_text }}</p>
        </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    mode = "course"
    error = None
    course_a = course_b = ""
    clo_a = clo_b = ""
    result = None
    level_results = None
    clo_sim = None
    clo_a_text = clo_b_text = ""
    clo_a_course = clo_b_course = ""

    if request.method == "POST":
        mode = request.form.get("mode", "course")

        if mode in ["course", "course_level"]:
            course_a = request.form.get("course_a", "").strip()

            if not course_a:
                error = "Please select Course A."
            elif mode == "course":
                course_b = request.form.get("course_b", "").strip()
                if not course_b:
                    error = "Please select Course B."
                elif course_a == course_b:
                    error = "Course A and Course B must be different."
                else:
                    sim_info = course_vs_course_similarity(course_a, course_b)
                    if sim_info is None:
                        error = "Could not find CLOs for one or both selected courses."
                    else:
                        for row in sim_info["details"]:
                            row["sim_class"] = sim_class(row["similarity"])
                        result = sim_info
            else:  # course_level
                level_results = course_vs_level_similarity(course_a)
                for row in level_results:
                    row["sim_class"] = sim_class(row["overall"])

        elif mode == "clo":
            clo_a = request.form.get("clo_a", "").strip()
            clo_b = request.form.get("clo_b", "").strip()
            if not clo_a or not clo_b:
                error = "Please select both CLO A and CLO B."
            elif clo_a == clo_b:
                error = "CLO A and CLO B must be different."
            else:
                try:
                    idx_a = int(clo_a)
                    idx_b = int(clo_b)
                    row_a = df.loc[idx_a]
                    row_b = df.loc[idx_b]
                except Exception:
                    error = "Invalid CLO selection."
                else:
                    clo_a_text = row_a["CLO_TEXT"]
                    clo_b_text = row_b["CLO_TEXT"]
                    clo_a_course = row_a["COURSE"]
                    clo_b_course = row_b["COURSE"]
                    clo_sim = cosine_sim_rows(idx_a, idx_b)

    overall_class = sim_class(result["overall"]) if result else ""
    clo_sim_class = sim_class(clo_sim) if clo_sim is not None else ""

    return render_template_string(
        TEMPLATE,
        mode=mode,
        courses=COURSE_LIST,
        clos=CLO_OPTIONS,
        course_a=course_a,
        course_b=course_b,
        clo_a=clo_a,
        clo_b=clo_b,
        result=result,
        level_results=level_results,
        overall_class=overall_class,
        clo_sim=clo_sim,
        clo_sim_class=clo_sim_class,
        clo_a_text=clo_a_text,
        clo_b_text=clo_b_text,
        clo_a_course=clo_a_course,
        clo_b_course=clo_b_course,
        error=error,
        data_file=DATA_FILE,
        sheet_name=SHEET_NAME,
    )


if __name__ == "__main__":
    # debug=False to avoid reloading twice; leave True if you want auto-reload.
    app.run(debug=False)
