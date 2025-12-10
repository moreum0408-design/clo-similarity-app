import os
import re
from collections import defaultdict, OrderedDict

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

DATA_FILE = "All CLOs copy.xlsx"
SHEET_NAME = "All CLOs_data (1)"

DISPLAY_TOP = 25        # target number of grouped rows
HASH_DIM = 4096         # hashing dimension (keeps memory smaller)


def parse_course_level(code):
    """Return 100, 200, 300, ... from the first digit of the course number."""
    if not isinstance(code, str):
        code = str(code)
    m = re.search(r"(\d+)", code)
    if not m:
        return None
    return int(m.group(1)[0]) * 100


def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found")

    print("Reading Excel...")
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME).reset_index(drop=True)

    # detect course column
    if "COURSE" in df.columns:
        course_col = "COURSE"
    elif "Course" in df.columns:
        course_col = "Course"
    else:
        raise ValueError("No COURSE or Course column in Excel")

    df["CLO_TEXT"] = df["CLO_TEXT"].fillna("").astype(str)
    df[course_col] = df[course_col].astype(str)

    n = len(df)
    print(f"{n} CLO rows.")

    # -------- Vectorize CLO text --------
    print("Vectorizing text with HashingVectorizer...")
    hv = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=False,
        norm=None,
    )
    X = hv.transform(df["CLO_TEXT"])
    X = normalize(X, axis=1)  # row-normed vectors

    # map: course -> row indices
    course_to_rows = defaultdict(list)
    for i, c in df[course_col].items():
        course_to_rows[c].append(i)

    courses = sorted(course_to_rows.keys())

    # map: level -> list of courses
    level_to_courses = defaultdict(list)
    for c in courses:
        lvl = parse_course_level(c)
        if lvl is not None:
            level_to_courses[lvl].append(c)

    # =========================================================
    #  COURSE vs COURSE  (grouped, up to 25 similarity groups)
    # =========================================================
    print("Computing course–course similarities (grouped to 25)...")
    course_rows = []

    for course_a in courses:
        lvl_a = parse_course_level(course_a)
        idx_a = course_to_rows[course_a]
        if not idx_a or lvl_a is None:
            continue

        same_level_courses = [
            c for c in level_to_courses[lvl_a] if c != course_a
        ]
        if not same_level_courses:
            continue

        mean_a = X[idx_a].mean(axis=0)
        mean_a_vec = np.asarray(mean_a).ravel()

        sims = []
        for course_b in same_level_courses:
            idx_b = course_to_rows[course_b]
            if not idx_b:
                continue
            mean_b = X[idx_b].mean(axis=0)
            mean_b_vec = np.asarray(mean_b).ravel()
            sim = float(mean_a_vec.dot(mean_b_vec))
            sims.append((course_b, sim))

        if not sims:
            continue

        sims.sort(key=lambda x: x[1], reverse=True)

        grouped = OrderedDict()  # key = rounded sim -> list of (course_b, sim)

        for course_b, sim in sims:
            key = round(sim, 3)  # group by similarity to 3 decimals

            if key in grouped:
                # existing group, always allowed
                grouped[key].append((course_b, sim))
            else:
                # new group – only create if we don't already have 25 groups
                if len(grouped) >= DISPLAY_TOP:
                    # we already have 25 groups; ignore new keys
                    continue
                grouped[key] = [(course_b, sim)]

        # flatten out into rows
        for key, entries in grouped.items():
            for course_b, sim in entries:
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

    # =========================================================
    #  CLO vs CLO  (grouped by CLO text, up to 25 distinct texts)
    # =========================================================
    print("Computing CLO–CLO similarities (grouped to 25 texts)...")
    clo_rows = []

    for base_idx in range(n):
        if base_idx % 1000 == 0:
            print(f"  CLO {base_idx}/{n}")

        base_course = df.at[base_idx, course_col]
        lvl = parse_course_level(base_course)
        if lvl is None:
            continue

        # candidates: all CLOs in same-level courses, EXCEPT any CLOs in the same course
        same_level_courses = [
            c for c in level_to_courses[lvl] if c != base_course
        ]
        if not same_level_courses:
            continue

        cand_indices = []
        for c in same_level_courses:
            cand_indices.extend(course_to_rows[c])

        if not cand_indices:
            continue

        base_vec = X[base_idx]       # 1 x D
        cand_mat = X[cand_indices]   # M x D

        sims_sparse = base_vec.dot(cand_mat.T)
        sims = np.asarray(sims_sparse.todense()).ravel()

        pairs = list(zip(cand_indices, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)

        grouped = OrderedDict()  # CLO_TEXT -> list of (other_idx, sim)

        for other_idx, sim in pairs:
            if other_idx == base_idx:
                continue

            clo_text = df.at[other_idx, "CLO_TEXT"]

            if clo_text in grouped:
                grouped[clo_text].append((other_idx, sim))
            else:
                if len(grouped) >= DISPLAY_TOP:
                    # already have 25 distinct CLO texts – ignore new ones
                    continue
                grouped[clo_text] = [(other_idx, sim)]

        for clo_text, entries in grouped.items():
            for other_idx, sim in entries:
                clo_rows.append(
                    {
                        "base_idx": base_idx,
                        "other_idx": int(other_idx),
                        "similarity": float(sim),
                    }
                )

    clo_sim_df = pd.DataFrame(clo_rows)
    clo_sim_df.to_csv("clo_similarity_top.csv", index=False)
    print(f"Saved clo_similarity_top.csv ({len(clo_sim_df)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
