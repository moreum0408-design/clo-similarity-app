import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

TARGET_COURSE_ROWS = 25   # how many course-vs-course rows we keep per course
TARGET_CLO_ROWS = 20      # how many CLO-vs-CLO rows we keep per base CLO
HASH_DIM = 4096           # hashing dimension for text vectors (keeps RAM down)


def parse_course_level(code):
    if not isinstance(code, str):
        code = str(code)

    m = re.search(r"(\d+)", code)
    if not m:
        return None

    num = int(m.group(1))
    # floor to the nearest hundred using the FULL number:
    # 338  -> (338 // 100) * 100  = 3 * 100   = 300
    # 3001 -> (3001 // 100) * 100 = 30 * 100  = 3000
    # 8888 -> 8800, etc.
    return (num // 100) * 100


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in current folder.")

    print("Reading Excel...")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    # Figure out course column name
    if "COURSE" in df.columns:
        course_col = "COURSE"
    elif "Course" in df.columns:
        course_col = "Course"
    else:
        raise ValueError("No COURSE or Course column found in Excel.")

    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].astype(str)

    n = len(df)
    print(f"{n} CLO rows.")

    # ---------- Build text vectors once ----------
    print("Vectorizing CLO text with HashingVectorizer...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(df["CLO_TEXT"])
    X = normalize(X, axis=1)
    print("Vectorization done.")

    # ---------- Per-course bookkeeping ----------
    course_to_rows = defaultdict(list)
    for i, c in df[course_col].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())
    print(f"{len(courses)} distinct courses.")

    print("Precomputing course levels and mean vectors...")
    course_level = {}
    course_vec = {}
    for c in courses:
        lvl = parse_course_level(c)
        course_level[c] = lvl
        idxs = course_to_rows[c]
        if not idxs or lvl is None:
            continue
        mean = X[idxs].mean(axis=0)
        course_vec[c] = np.asarray(mean).ravel()

    # ---------- COURSE vs COURSE (top 25 per course) ----------
    print("Computing course-vs-course similarities...")
    course_rows = []
    for i, course_a in enumerate(courses):
        lvl_a = course_level.get(course_a)
        vec_a = course_vec.get(course_a)
        if lvl_a is None or vec_a is None:
            continue

        if i % 200 == 0:
            print(f"  Course {i}/{len(courses)}: {course_a}")

        sims = []
        for course_b in courses:
            if course_b == course_a:
                continue
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

    # ---------- CLO vs CLO (up to 20 UNIQUE per base CLO) ----------
    print("Precomputing per-row levels...")
    # cache level per row so we don't keep reparsing
    df["LEVEL"] = df[course_col].map(
        lambda c: course_level.get(c, parse_course_level(c))
    )
    levels = df["LEVEL"].to_numpy()
    course_arr = df[course_col].to_numpy()

    clo_rows = []
    print("Computing CLO-vs-CLO similarities (filling up to 20 rows with same-level CLOs)...")

    for base_idx in range(n):
        if base_idx % 2000 == 0:
            print(f"  CLO {base_idx}/{n}")

        base_level = levels[base_idx]
        if base_level is None:
            continue

        base_course = course_arr[base_idx]
        base_vec = X[base_idx]

        # Candidates = ALL CLOs at the same level, but NOT the same course
        same_level_mask = (levels == base_level) & (course_arr != base_course)
        cand_indices = np.where(same_level_mask)[0]
        if cand_indices.size == 0:
            continue

        cand_mat = X[cand_indices]          # M x D
        sims_sparse = base_vec.dot(cand_mat.T)
        sims = np.asarray(sims_sparse.todense()).ravel()   # length M

        # Sort candidates by similarity, highest first
        order = np.argsort(-sims)

        # Deduplicate by (course, CLO_TEXT), keep the best similarity for each
        dedup = {}
        for pos in order:
            idx_other = int(cand_indices[pos])
            sim = float(sims[pos])

            course_b = course_arr[idx_other]
            text_b = df.at[idx_other, "CLO_TEXT"]
            key = (course_b, text_b)

            if key in dedup:
                continue

            dedup[key] = (idx_other, sim)
            if len(dedup) >= TARGET_CLO_ROWS:
                break

        # Now write out up to TARGET_CLO_ROWS rows, sorted by similarity
        for idx_other, sim in sorted(
            dedup.values(), key=lambda x: x[1], reverse=True
        ):
            clo_rows.append(
                {
                    "base_idx": base_idx,
                    "other_idx": int(idx_other),
                    "similarity": sim,
                }
            )

    clo_sim_df = pd.DataFrame(clo_rows)
    clo_sim_df.to_csv("clo_similarity_top.csv", index=False)
    print(f"Saved clo_similarity_top.csv ({len(clo_sim_df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
