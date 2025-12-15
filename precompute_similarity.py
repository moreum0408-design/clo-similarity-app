import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

# =============================
# CONFIG
# =============================
DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

TARGET_COURSE_ROWS = 25   # per base course
TARGET_CLO_ROWS = 20      # per base CLO
HASH_DIM = 4096

# HARD FILTERS
MIN_SIMILARITY = 0.60     # anything below is dropped (no exceptions)

# =============================
# NORMALIZATION & PARSING
# =============================
def normalize_course(code):
    """
    Normalize course codes so parsing is deterministic.
    Examples:
      ' HLSC 302 '   -> 'HLSC302'
      'lcom-3020'    -> 'LCOM3020'
      'CJUS 338A'    -> 'CJUS338A' (digits still parsed correctly)
    """
    if not isinstance(code, str):
        code = str(code)
    code = code.upper().strip()
    # keep only letters and digits
    code = re.sub(r"[^A-Z0-9]", "", code)
    return code


def extract_digits(code):
    """Return the numeric string inside a course code, or None."""
    m = re.search(r"(\d+)", code)
    return m.group(1) if m else None


def digit_count(code):
    """3 for 3xx, 4 for 30xx, etc."""
    d = extract_digits(code)
    return len(d) if d else None


def level_bucket(code):
    """
    STRICT bucket:
      302  -> 300
      338  -> 300
      3020 -> 3000
      3001 -> 3000
    """
    d = extract_digits(code)
    if not d:
        return None
    num = int(d)
    return (num // 100) * 100


# =============================
# MAIN
# =============================
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
        raise ValueError("No COURSE/Course column found")

    if "CLO_TEXT" not in df.columns:
        raise ValueError("No CLO_TEXT column found")

    # Clean inputs
    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].fillna("").astype(str)

    # Normalize course codes ONCE
    df["COURSE_NORM"] = df[course_col].map(normalize_course)
    df["DIGITS"] = df["COURSE_NORM"].map(digit_count)
    df["LEVEL"] = df["COURSE_NORM"].map(level_bucket)

    n = len(df)
    print(f"{n} CLO rows after normalization")

    # =============================
    # VECTORIZE TEXT
    # =============================
    print("Vectorizing CLO text...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(df["CLO_TEXT"])
    X = normalize(X, axis=1)
    print("Vectorization done")

    # =============================
    # GROUP BY COURSE
    # =============================
    course_to_rows = defaultdict(list)
    for i, c in df["COURSE_NORM"].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())
    print(f"{len(courses)} distinct normalized courses")

    # Precompute course vectors
    course_vec = {}
    course_level = {}
    course_digits = {}

    for c in courses:
        lvl = level_bucket(c)
        dct = digit_count(c)
        course_level[c] = lvl
        course_digits[c] = dct

        idxs = course_to_rows[c]
        if not idxs or lvl is None or dct is None:
            continue

        mean_vec = X[idxs].mean(axis=0)
        course_vec[c] = np.asarray(mean_vec).ravel()

    # =============================
    # COURSE vs COURSE (STRICT)
    # =============================
    print("Computing course-vs-course (STRICT digit + level)...")
    course_rows = []

    for course_a in courses:
        lvl_a = course_level.get(course_a)
        dig_a = course_digits.get(course_a)
        vec_a = course_vec.get(course_a)
        if lvl_a is None or dig_a is None or vec_a is None:
            continue

        sims = []
        for course_b in courses:
            if course_b == course_a:
                continue

            # HARD GATES (NO FALLBACK)
            if course_level.get(course_b) != lvl_a:
                continue
            if course_digits.get(course_b) != dig_a:
                continue

            vec_b = course_vec.get(course_b)
            if vec_b is None:
                continue

            sim = float(vec_a @ vec_b)
            if sim < MIN_SIMILARITY:
                continue

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

    # =============================
    # CLO vs CLO (STRICT)
    # =============================
    print("Computing CLO-vs-CLO (STRICT digit + level, no fallback)...")
    clo_rows = []

    levels = df["LEVEL"].to_numpy()
    digits = df["DIGITS"].to_numpy()
    courses_arr = df["COURSE_NORM"].to_numpy()

    for base_idx in range(n):
        base_level = levels[base_idx]
        base_digits = digits[base_idx]
        if base_level is None or base_digits is None:
            continue

        base_course = courses_arr[base_idx]
        base_vec = X[base_idx]

        # HARD MASK: same level + same digit count + different course
        mask = (
            (levels == base_level)
            & (digits == base_digits)
            & (courses_arr != base_course)
        )
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

            if sim < MIN_SIMILARITY:
                break  # because sorted desc

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

    print("DONE — STRICT 3xx vs 30xx, digit-count enforced, threshold applied")


if __name__ == "__main__":
    main()
