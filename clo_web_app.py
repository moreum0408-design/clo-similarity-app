import os
from textwrap import shorten

from flask import Flask, request, render_template_string
import pandas as pd

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

COURSE_SIM_FILE = "course_similarity_top.csv"
CLO_SIM_FILE = "clo_similarity_top.csv"

# ---------- Load Excel ----------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"{DATA_FILE} not found in project folder.")

df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

if "COURSE" in df.columns:
    COURSE_COL = "COURSE"
elif "Course" in df.columns:
    COURSE_COL = "Course"
else:
    raise ValueError("No COURSE or Course column in Excel")

df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
df[COURSE_COL] = df[COURSE_COL].astype(str)

COURSES = sorted(df[COURSE_COL].unique())

# ---------- Load precomputed similarity ----------
if not (os.path.exists(COURSE_SIM_FILE) and os.path.exists(CLO_SIM_FILE)):
    raise FileNotFoundError(
        "Missing course_similarity_top.csv or clo_similarity_top.csv. "
        "Run precompute_similarity.py locally and add the CSVs."
    )

course_sim_df = pd.read_csv(COURSE_SIM_FILE)
clo_sim_df = pd.read_csv(CLO_SIM_FILE)


# ---------- Helpers ----------
def course_clo_options(course):
    if not course:
        return []
    subset = df[df[COURSE_COL] == course].copy()
    subset = subset.drop_duplicates(subset=["CLO_TEXT"])
    opts = []
    for idx in subset.index:
        label = shorten(subset.at[idx, "CLO_TEXT"], width=120, placeholder="…")
        opts.append({"id": int(idx), "label": label})
    return opts


def course_vs_level_from_csv(course_a):
    subset = course_sim_df[course_sim_df["base_course"] == course_a]
    subset = subset.sort_values("overall", ascending=False)
    return [
        {"course_b": row["other_course"], "overall": float(row["overall"])}
        for _, row in subset.iterrows()
    ]


def clo_vs_level_from_csv(base_idx):
    subset = clo_sim_df[clo_sim_df["base_idx"] == base_idx]
    subset = subset.sort_values("similarity", ascending=False)
    results = []
    for _, row in subset.iterrows():
        other_idx = int(row["other_idx"])
        results.append(
            {
                "course": df.loc[other_idx, COURSE_COL],
                "clo_text": df.loc[other_idx, "CLO_TEXT"],
                "similarity": float(row["similarity"]),
            }
        )
    return results


# ---------- Flask app ----------
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
    <h2>Overall Course vs Same-Level Courses (top {{ course_level_results|length }})</h2>
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
    <h2>This CLO vs CLOs in Same-Level Courses (top {{ clo_results|length }})</h2>
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

    course_clos = course_clo_options(course_a)

    if request.method == "POST":
        if not course_a:
            error = "Select a course."
        elif not clo_idx:
            pass
        else:
            try:
                base_idx = int(clo_idx)
                if base_idx < 0 or base_idx >= len(df):
                    error = "Invalid CLO selection."
                elif df.loc[base_idx, COURSE_COL] != course_a:
                    error = "Selected CLO does not belong to that course."
                else:
                    base_clo_text = df.loc[base_idx, "CLO_TEXT"]
                    course_level_results = course_vs_level_from_csv(course_a)
                    clo_results = clo_vs_level_from_csv(base_idx)
            except Exception as e:
                error = f"Invalid CLO selection: {e}"

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
