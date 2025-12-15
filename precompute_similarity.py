import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

TARGET_COURSE_ROWS = 25
TARGET_CLO_ROWS = 20
HASH_DIM = 4096


# -----------------------------
# COURSE NORMALIZATION
# -----------------------------
def normalize_course(code):
    """
    Normalize course codes so parsing is deterministic.
    Examples:
      ' HLSC 302 '   -> 'HLSC302'
      'lcom3020'     -> 'LCOM3020'
      'CJUS-338'     -> 'CJUS338'
    """
    if not isinstance(code, str):
        code = str(code)

    code = code.upper().strip()
    code = re.sub(r"[^A-Z0-9]", "", code)
    return code


def parse_course_level(code):
    """
    STRICT level bucketing:
      302   -> 300
      338   -> 300
      3020  -> 3000
      3001  -> 3000
    """
    m = re.search(r"(\d+)", code)
    if not m:
        return None

    num = int(m.group(1))
    return (num // 100) * 100


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    print("Reading Excel...")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    # Identify course column
    if "COURSE" in df.columns:
        course_col = "COURSE"
    elif "Course" in df.columns:
        course_col = "Course"
    else:
        raise ValueError("No COURSE column found")

    if "CLO_TEXT" not in df.columns:
        raise ValueError("No CLO_TEXT column found")

    # -----------------------------
    # NORMALIZE INPUT
    # -----------------------------
    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].fillna("").astype(str)

    # Normalize course codes ONCE
    df["COURSE_NORM"] = df[course_col].map(normalize_course)

    # Precompute levels ONCE
    df["LEVEL"] = df["COURSE_NORM"].map(parse_course_level)

    n = len(df)
    print(f"{n} CLO rows after normalization")

    # -----------------------------
    # VECTORIZE TEXT
    # -----------------------------
    print("Vectorizing CLO text...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(df["CLO_TEXT"])
    X = normalize(X, axis=1)
    print("Vectorization complete")

    # -----------------------------
    # COURSE GROUPING
    # -----------------------------
    course_to_rows = defaultdict(list)
    for i, c in df["COURSE_NORM"].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())
    print(f"{len(courses)} distinct normalized courses")

    # -----------------------------
    # COURSE MEAN VECTORS
    # -----------------------------
    course_level = {}
    course_vec = {}

    for c in courses:
        lvl = parse_course_level(c)
        course_level[c] = lvl

        idxs = course_to_rows[c]
        if not idxs or lvl is None:
            continue

        mean_vec = X[idxs].mean(axis=0)
        course_vec[c] = np.asarray(mean_vec).ravel()

    # -----------------------------
    # COURSE vs COURSE (STRICT LEVEL)
    # -----------------------------
    print("Computing course-vs-course similarities...")
    course_rows = []

    for course_a in courses:
        lvl_a = course_level.get(course_a)
        vec_a = course_vec.get(course_a)
        if lvl_a is None or vec_a is None:
            continue

        sims = []
        for course_b in courses:
            if course_b == course_a:
                continue

            # STRICT: must be exact same bucket (300 != 3000)
            if course_level.get(course_b) != lvl_a:
                continue

            vec_b = course_vec.get(course_b)
            if vec_b is None:
                continue

            sim = float(vec_a @ vec_b)
            sims.append((course_b, sim))

        sims.sort(key=lambda x: x[1], reverse=True)

        for course_b, sim in sims[:TARGET_COURSE_ROWS]:
            course_rows.append(
                {
                    "base_course": course_a,
                    "other_course": course_b,
                    "overall": sim,
                }
            )

    course_sim_df = pd.DataFrame(course_rows)
    course_sim_df.to_csv("course_similarity_top.csv", index=False)
    print(f"Saved course_similarity_top.csv ({len(course_sim_df)} rows)")

    # -----------------------------
    # CLO vs CLO (STRICT LEVEL)
    # -----------------------------
    print("Computing CLO-vs-CLO similarities...")
    clo_rows = []

    levels = df["LEVEL"].to_numpy()
    courses_arr = df["COURSE_NORM"].to_numpy()

    for base_idx in range(n):
        base_level = levels[base_idx]
        if base_level is None:
            continue

        base_course = courses_arr[base_idx]
        base_vec = X[base_idx]

        # STRICT: same level AND different course
        mask = (levels == base_level) & (courses_arr != base_course)
        cand_indices = np.where(mask)[0]

        if cand_indices.size == 0:
            continue

        cand_mat = X[cand_indices]
        sims = np.asarray(base_vec.dot(cand_mat.T).todense()).ravel()
        order = np.argsort(-sims)

        dedup = {}
        for pos in order:
            idx_other = int(cand_indices[pos])
            sim = float(sims[pos])

            key = (courses_arr[idx_other], df.at[idx_other, "CLO_TEXT"])
            if key in dedup:
                continue

            dedup[key] = (idx_other, sim)
            if len(dedup) >= TARGET_CLO_ROWS:
                break

        for idx_other, sim in sorted(dedup.values(), key=lambda x: x[1], reverse=True):
            clo_rows.append(
                {
                    "base_idx": base_idx,
                    "other_idx": idx_other,
                    "similarity": sim,
                }
            )

    clo_sim_df = pd.DataFrame(clo_rows)
    clo_sim_df.to_csv("clo_similarity_top.csv", index=False)
    print(f"Saved clo_similarity_top.csv ({len(clo_sim_df)} rows)")

    print("DONE — strict 3xx vs 30xx enforced")


if __name__ == "__main__":
    main()
