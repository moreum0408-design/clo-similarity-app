import os
from textwrap import shorten
from collections import defaultdict

from flask import Flask, request, render_template_string, url_for
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

COURSE_SIM_FILE = "course_similarity_top.csv"
CLO_SIM_FILE = "clo_similarity_top.csv"

DISPLAY_TOP = 25
HASH_DIM = 4096  # for on-the-fly CLO similarity


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
    """All distinct CLOs in a course (for dropdown)."""
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
    """Use precomputed data to build grouped course rows (up to 25)."""
    subset = course_sim_df[course_sim_df["base_course"] == course_a]
    subset = subset.sort_values("overall", ascending=False)

    raw = [
        {"course_b": row["other_course"], "overall": float(row["overall"])}
        for _, row in subset.iterrows()
    ]

    groups = defaultdict(list)  # rounded sim -> list of course_b
    for r in raw:
        key = round(r["overall"], 3)
        groups[key].append(r["course_b"])

    rows = []
    for sim in sorted(groups.keys(), reverse=True):
        courses_list = sorted(set(groups[sim]))
        rows.append(
            {
                "courses": " / ".join(courses_list),
                "courses_list": courses_list,
                "overall": sim,
            }
        )

    return rows[:DISPLAY_TOP]


def clo_vs_level_from_csv(base_idx):
    """Use precomputed data to build grouped CLO rows (up to 25)."""
    subset = clo_sim_df[clo_sim_df["base_idx"] == base_idx]
    subset = subset.sort_values("similarity", ascending=False)

    raw = []
    for _, row in subset.iterrows():
        other_idx = int(row["other_idx"])
        raw.append(
            {
                "course": df.loc[other_idx, COURSE_COL],
                "clo_text": df.loc[other_idx, "CLO_TEXT"],
                "similarity": float(row["similarity"]),
            }
        )

    # group by CLO text; collect all courses that share that text
    groups = {}
    for r in raw:
        text = r["clo_text"]
        sim = r["similarity"]
        if text not in groups:
            groups[text] = {"courses": set(), "similarity": sim}
        groups[text]["courses"].add(r["course"])
        if sim > groups[text]["similarity"]:
            groups[text]["similarity"] = sim

    rows = []
    for text, info in groups.items():
        courses_list = sorted(info["courses"])
        rows.append(
            {
                "courses": " / ".join(courses_list),
                "courses_list": courses_list,
                "clo_text": text,
                "similarity": info["similarity"],
            }
        )

    rows.sort(key=lambda x: x["similarity"], reverse=True)
    return rows[:DISPLAY_TOP]


def compute_clo_detail_similarities(base_idx, target_course):
    """
    For a clicked course, compute similarity of the selected base CLO
    against ALL CLOs in the target course (on the fly, same level).
    """
    base_idx = int(base_idx)
    if base_idx < 0 or base_idx >= len(df):
        raise ValueError("Invalid base CLO index.")

    base_text = df.loc[base_idx, "CLO_TEXT"]
    base_course = df.loc[base_idx, COURSE_COL]

    subset = df[df[COURSE_COL] == target_course].copy()
    if subset.empty:
        return base_course, base_text, []

    texts = [base_text] + subset["CLO_TEXT"].tolist()

    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X_local = hv.transform(texts)
    X_local = normalize(X_local, axis=1)

    base_vec = X_local[0]
    cand_mat = X_local[1:]
    sims_sparse = base_vec.dot(cand_mat.T)
    sims = sims_sparse.toarray().ravel()

    results = []
    for (idx, (_, row)), sim in zip(
        enumerate(subset.itertuples(index=True)), sims
    ):
        results.append(
            {
                "clo_text": row.CLO_TEXT,
                "similarity": float(sim),
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return base_course, base_text, results


# ---------- Flask app ----------
app = Flask(__name__)

MAIN_HTML = """<!doctype html>
<html>
<head>
<title>CLO Similarity Tool</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
.result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
table { border-collapse: collapse; width: 100%; margin-top: 10px;}
th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
.sim { font-weight:bold; }
a.course-link { text-decoration: none; }
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
    <p><strong>Base Course:</strong> {{ course_a }}</p>
    <p><strong>Base CLO:</strong> {{ base_clo_text }}</p>
    <p><em>Tip:</em> Click on any course code below to see <strong>all CLOs</strong> in that course and their similarity to this CLO.</p>
</div>
{% endif %}

{% if course_level_results %}
<div class="result">
    <h2>Overall Course vs Same-Level Courses (grouped, top {{ course_level_results|length }})</h2>
    <table>
        <tr><th>Course(s)</th><th>Overall Similarity</th></tr>
        {% for r in course_level_results %}
        <tr>
            <td>
                {% for cc in r.courses_list %}
                    <a class="course-link"
                       href="{{ url_for('course_detail', course_a=course_a, clo_idx=clo_idx, target_course=cc) }}"
                       target="_blank">{{ cc }}</a>{% if not loop.last %} / {% endif %}
                {% endfor %}
            </td>
            <td class="sim">{{ r.overall|round(3) }}</td>
        </tr>
        {% endfor %}
    </table>
</div>
{% endif %}

{% if clo_results %}
<div class="result">
    <h2>This CLO vs CLOs in Same-Level Courses (grouped, top {{ clo_results|length }})</h2>
    <table>
        <tr><th>Course(s)</th><th>CLO</th><th>Similarity</th></tr>
        {% for r in clo_results %}
        <tr>
            <td>
                {% for cc in r.courses_list %}
                    <a class="course-link"
                       href="{{ url_for('course_detail', course_a=course_a, clo_idx=clo_idx, target_course=cc) }}"
                       target="_blank">{{ cc }}</a>{% if not loop.last %} / {% endif %}
                {% endfor %}
            </td>
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


DETAIL_HTML = """<!doctype html>
<html>
<head>
<title>CLO Detail Similarity</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px;}
th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
.sim { font-weight:bold; }
.result { margin-top: 20px; padding: 15px; background: #f5f5f5; }
a { text-decoration: none; }
</style>
</head>
<body>
<h1>CLO Detail Similarity</h1>

<div class="result">
    <p><strong>Base Course:</strong> {{ base_course }}</p>
    <p><strong>Base CLO:</strong> {{ base_clo_text }}</p>
    <p><strong>Target Course:</strong> {{ target_course }}</p>
</div>

{% if clo_sims %}
<table>
    <tr><th>Target Course CLO</th><th>Similarity to Base CLO</th></tr>
    {% for r in clo_sims %}
    <tr>
        <td>{{ r.clo_text }}</td>
        <td class="sim">{{ r.similarity|round(3) }}</td>
    </tr>
    {% endfor %}
</table>
{% else %}
<p>No CLOs found for this target course.</p>
{% endif %}

<p><a href="{{ url_for('index') }}">Back to main tool</a></p>

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
            # just refresh CLO dropdown
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
        MAIN_HTML,
        courses=COURSES,
        course_a=course_a,
        clo_idx=clo_idx,
        course_clos=course_clos,
        error=error,
        base_clo_text=base_clo_text,
        course_level_results=course_level_results,
        clo_results=clo_results,
    )


@app.route("/course_detail")
def course_detail():
    course_a = request.args.get("course_a", "")
    clo_idx = request.args.get("clo_idx", "")
    target_course = request.args.get("target_course", "")

    try:
        base_idx = int(clo_idx)
    except Exception:
        return "Invalid CLO selection.", 400

    try:
        base_course, base_clo_text, clo_sims = compute_clo_detail_similarities(
            base_idx, target_course
        )
    except Exception as e:
        return f"Error computing details: {e}", 500

    return render_template_string(
        DETAIL_HTML,
        base_course=base_course,
        base_clo_text=base_clo_text,
        target_course=target_course,
        clo_sims=clo_sims,
    )


if __name__ == "__main__":
    app.run(debug=False)
