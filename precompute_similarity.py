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


def parse_course_level(code: str):
    if not isinstance(code, str):
        return None
    code = code.strip()

    m = re.search(r"(\d+)", code)
    if not m:
        return None

    digits_str = m.group(1)
    num = int(digits_str)
    digits = len(digits_str)

    # STRICT: 3-digit ≠ 4-digit
    # 300–399 -> 300
    # 3000–3999 -> 3000
    if digits == 3:
        return (num // 100) * 100
    if digits == 4:
        return (num // 1000) * 1000

    return None


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    print("Reading Excel...")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    course_col = "COURSE" if "COURSE" in df.columns else "Course"

    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].fillna("").astype(str).str.strip()

    n = len(df)
    print(f"{n} CLO rows after normalization")

    print("Vectorizing CLO text...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
        stop_words="english",
    )
    X = normalize(hv.transform(df["CLO_TEXT"]), axis=1)
    print("Vectorization done")

    print("Indexing courses...")
    course_to_rows = defaultdict(list)
    for i, c in df[course_col].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())
    print(f"{len(courses)} distinct normalized courses")

    print("Computing course levels + mean vectors...")
    course_level = {}
    course_vec = {}
    for c in courses:
        lvl = parse_course_level(c)
        course_level[c] = lvl
        if lvl is None:
            continue
        idxs = course_to_rows[c]
        if not idxs:
            continue
        mean_vec = X[idxs].mean(axis=0)
        course_vec[c] = np.asarray(mean_vec).ravel()

    # ===============================
    # COURSE vs COURSE (TOP 25)
    # ===============================
    print("Computing course-vs-course (STRICT digit + level)...")
    course_rows = []
    for c1 in courses:
        lvl1 = course_level.get(c1)
        v1 = course_vec.get(c1)
        if lvl1 is None or v1 is None:
            continue

        sims = []
        for c2 in courses:
            if c1 == c2:
                continue
            if course_level.get(c2) != lvl1:
                continue
            v2 = course_vec.get(c2)
            if v2 is None:
                continue
            sims.append((c2, float(v1 @ v2)))

        sims.sort(key=lambda x: x[1], reverse=True)
        for c2, s in sims[:TARGET_COURSE_ROWS]:
            course_rows.append(
                {"base_course": c1, "other_course": c2, "overall": s}
            )

    pd.DataFrame(course_rows).to_csv("course_similarity_top.csv", index=False)
    print(f"Saved course_similarity_top.csv ({len(course_rows)} rows)")

    # ===============================
    # CLO vs CLO (TOP 20 UNIQUE)
    # ===============================
    print("Computing CLO-vs-CLO (STRICT digit + level, TOP 20 UNIQUE)...")

    df["LEVEL"] = df[course_col].map(parse_course_level)
    levels = df["LEVEL"].to_numpy()
    course_arr = df[course_col].to_numpy()

    clo_rows = []

    for i in range(n):
        if i % 2000 == 0:
            print(f"  CLO {i}/{n}")

        base_lvl = levels[i]
        if base_lvl is None:
            continue

        base_course = course_arr[i]
        base_text = df.at[i, "CLO_TEXT"]
        base_vec = X[i]

        # same level, different course
        mask = (levels == base_lvl) & (course_arr != base_course)
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            continue

        sims = base_vec.dot(X[cand_idx].T).toarray().ravel()
        order = np.argsort(-sims)

        # Dedup by (compare_course, compare_clo)
        seen = set()
        picked = 0

        for pos in order:
            j = int(cand_idx[pos])
            compare_course = course_arr[j]
            compare_clo = df.at[j, "CLO_TEXT"]
            key = (compare_course, compare_clo)
            if key in seen:
                continue
            seen.add(key)

            clo_rows.append(
                {
                    "base_course": base_course,
                    "base_clo": base_text,
                    "compare_course": compare_course,
                    "compare_clo": compare_clo,
                    "similarity": float(sims[pos]),
                }
            )

            picked += 1
            if picked >= TARGET_CLO_ROWS:
                break

    pd.DataFrame(clo_rows).to_csv("clo_similarity_top.csv", index=False)
    print(f"Saved clo_similarity_top.csv ({len(clo_rows)} rows)")
    print("DONE — strict 3xx vs 30xx, TOP 20 UNIQUE")


if __name__ == "__main__":
    main()
