import os
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"   # 너 precompute에 있던 시트명 그대로
CLO_SIM_FILE = "clo_similarity_top.csv"
COURSE_SIM_FILE = "course_similarity_top.csv"

TOP_COURSE_ROWS = 20
TOP_CLO_ROWS = 20


def _detect_course_col(df: pd.DataFrame) -> str:
    if "COURSE" in df.columns:
        return "COURSE"
    if "Course" in df.columns:
        return "Course"
    raise ValueError("Excel에서 COURSE 또는 Course 컬럼을 찾을 수 없음")


def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in current folder")
    if not os.path.exists(CLO_SIM_FILE):
        raise FileNotFoundError(f"{CLO_SIM_FILE} not found in current folder")
    if not os.path.exists(COURSE_SIM_FILE):
        raise FileNotFoundError(f"{COURSE_SIM_FILE} not found in current folder")

    df_all = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)
    course_col = _detect_course_col(df_all)

    # normalize
    df_all[course_col] = df_all[course_col].astype(str).str.strip()
    df_all["CLO_TEXT"] = df_all["CLO_TEXT"].fillna("").astype(str).str.strip()

    df_clo = pd.read_csv(CLO_SIM_FILE)
    df_course = pd.read_csv(COURSE_SIM_FILE)

    # sanity columns
    for col in ["base_idx", "other_idx", "similarity"]:
        if col not in df_clo.columns:
            raise ValueError(f"{CLO_SIM_FILE} missing column: {col}")

    for col in ["base_course", "other_course", "overall"]:
        if col not in df_course.columns:
            raise ValueError(f"{COURSE_SIM_FILE} missing column: {col}")

    return df_all, course_col, df_clo, df_course


# Render에서 worker마다 재로딩 안 하게 전역 로드
DF_ALL, COURSE_COL, DF_CLO_SIM, DF_COURSE_SIM = load_data()


@app.route("/", methods=["GET", "POST"])
def index():
    courses = sorted(DF_ALL[COURSE_COL].dropna().unique().tolist())

    selected_course = None
    selected_clo = None
    clo_options = []

    course_results = []
    clo_results = []

    if request.method == "POST":
        selected_course = (request.form.get("course") or "").strip()
        selected_clo = (request.form.get("clo") or "").strip()

        if selected_course:
            clo_options = (
                DF_ALL[DF_ALL[COURSE_COL] == selected_course]["CLO_TEXT"]
                .drop_duplicates()
                .tolist()
            )

            # Course vs course top 20 (same-level already filtered by precompute)
            course_results = (
                DF_COURSE_SIM[DF_COURSE_SIM["base_course"] == selected_course]
                .sort_values("overall", ascending=False)
                .head(TOP_COURSE_ROWS)
                .to_dict(orient="records")
            )

        # CLO vs CLO top 20 (NO GROUPING, pure top 20)
        if selected_course and selected_clo:
            # base_idx 찾기
            matches = DF_ALL[
                (DF_ALL[COURSE_COL] == selected_course) &
                (DF_ALL["CLO_TEXT"] == selected_clo)
            ]
            if len(matches) == 0:
                # 선택 CLO가 못 찾는 경우(문자열 공백 등) 대비
                selected_clo = selected_clo.strip()
                matches = DF_ALL[
                    (DF_ALL[COURSE_COL] == selected_course) &
                    (DF_ALL["CLO_TEXT"].str.strip() == selected_clo)
                ]

            if len(matches) > 0:
                base_idx = int(matches.index[0])

                sub = DF_CLO_SIM[DF_CLO_SIM["base_idx"] == base_idx].copy()
                sub = sub.sort_values("similarity", ascending=False).head(TOP_CLO_ROWS)

                # other_idx를 DF_ALL에 조인해서 course/clo_text 가져오기
                other_rows = DF_ALL.loc[sub["other_idx"].astype(int), [COURSE_COL, "CLO_TEXT"]].reset_index(drop=True)
                sub = sub.reset_index(drop=True)

                sub["Course"] = other_rows[COURSE_COL].astype(str).values
                sub["CLO"] = other_rows["CLO_TEXT"].astype(str).values

                clo_results = sub[["Course", "CLO", "similarity"]].to_dict(orient="records")

    return render_template(
        "index.html",
        courses=courses,
        selected_course=selected_course,
        selected_clo=selected_clo,
        clo_options=clo_options,
        course_results=course_results,
        clo_results=clo_results,
        TOP_COURSE_ROWS=TOP_COURSE_ROWS,
        TOP_CLO_ROWS=TOP_CLO_ROWS,
    )


if __name__ == "__main__":
    # 로컬 테스트용 (Render는 gunicorn으로 실행)
    app.run(host="127.0.0.1", port=5000, debug=False)
