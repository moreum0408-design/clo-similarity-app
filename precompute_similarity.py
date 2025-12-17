import os
import re
import math
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_XLSX = os.path.join(BASE_DIR, "All CLOs copy.xlsx")

OUT_COURSE = os.path.join(BASE_DIR, "course_similarity_top.csv")
OUT_CLO_TOP = os.path.join(BASE_DIR, "clo_similarity_top.csv")
OUT_DRILL = os.path.join(BASE_DIR, "clo_similarity_drilldown.csv")


# -----------------------------
# Course level parsing (your rule)
# -----------------------------
# 3 digits => 3xx, 4 digits => 30xx (two-digit prefix)
_course_re = re.compile(r"^\s*([A-Za-z]+)\s*0*([0-9]{3,4})\s*$")

def parse_course(course: str):
    """
    Returns (prefix, digits, level_key)
    Examples:
      'CJUS 338' -> ('CJUS','338','3xx')
      'LCOM3001' -> ('LCOM','3001','30xx')
    """
    if course is None:
        return None, None, None
    s = str(course).strip().replace("-", "").replace("_", "")
    m = _course_re.match(s)
    if not m:
        # fallback: keep prefix+digits extraction naive
        letters = "".join([c for c in s if c.isalpha()])
        nums = "".join([c for c in s if c.isdigit()])
        if len(nums) not in (3, 4):
            return letters.upper() if letters else None, nums if nums else None, None
        prefix = letters.upper() if letters else None
        digits = nums
    else:
        prefix = m.group(1).upper()
        digits = m.group(2)

    if len(digits) == 3:
        level_key = f"{digits[0]}xx"     # 3xx
    else:
        level_key = f"{digits[:2]}xx"    # 30xx

    return prefix, digits, level_key


# -----------------------------
# Text embedding + cosine sim
# -----------------------------
def get_embedder():
    """
    Try sentence-transformers (better).
    Fallback: TF-IDF (still works).
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return ("st", model)
    except Exception:
        return ("tfidf", None)


def cosine_matrix(A: np.ndarray, B: np.ndarray):
    # cosine(A,B) = A_norm @ B_norm.T
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


# -----------------------------
# Main
# -----------------------------
def main():
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"Missing input file: {INPUT_XLSX}")

    # You may need to adjust sheet/column names if your Excel is different.
    df = pd.read_excel(INPUT_XLSX)

    # Try to auto-detect likely columns
    col_course = None
    col_clo = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if col_course is None and ("course" in lc):
            col_course = c
        if col_clo is None and (lc == "clo" or "learning outcome" in lc or "outcome" in lc):
            col_clo = c

    if col_course is None or col_clo is None:
        raise ValueError(
            f"Could not detect course/CLO columns in {INPUT_XLSX}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[[col_course, col_clo]].copy()
    df.columns = ["course", "clo"]

    df["course"] = df["course"].astype(str).str.strip()
    df["clo"] = df["clo"].astype(str).fillna("").str.strip()
    df = df[df["course"].ne("") & df["clo"].ne("")].copy()

    # Parse levels
    parsed = df["course"].apply(parse_course)
    df["prefix"] = parsed.apply(lambda x: x[0])
    df["digits"] = parsed.apply(lambda x: x[1])
    df["level_key"] = parsed.apply(lambda x: x[2])

    # Drop rows with unparseable course level
    df = df[df["level_key"].notna()].copy()

    # Group CLOs per course
    course_to_clos = {}
    for course, g in df.groupby("course"):
        clos = g["clo"].drop_duplicates().tolist()
        course_to_clos[course] = clos

    # Courses per level
    level_to_courses = {}
    for course, g in df.groupby("course"):
        lvl = g["level_key"].iloc[0]
        level_to_courses.setdefault(lvl, []).append(course)

    # Build embeddings for all unique CLO strings
    all_clos = df["clo"].drop_duplicates().tolist()

    emb_type, model = get_embedder()
    if emb_type == "st":
        # Sentence-transformers
        print("Using sentence-transformers embeddings")
        clo_emb = model.encode(all_clos, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        clo_emb = clo_emb.astype(np.float32)
    else:
        # TF-IDF fallback
        print("Using TF-IDF embeddings (fallback)")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        X = vectorizer.fit_transform(all_clos)
        clo_emb = X.astype(np.float32)  # sparse
        # For sparse, cosine needs different handling. We'll handle per-query via dot + norms.

    clo_to_idx = {t: i for i, t in enumerate(all_clos)}

    # Helper: similarity between one CLO and list of CLOs
    def sim_one_to_many(base_clo_text: str, other_clo_texts: list[str]) -> np.ndarray:
        if emb_type == "st":
            i = clo_to_idx[base_clo_text]
            base_vec = clo_emb[i:i+1, :]
            other_idx = [clo_to_idx[t] for t in other_clo_texts]
            other_mat = clo_emb[other_idx, :]
            sims = (base_vec @ other_mat.T).reshape(-1)  # already normalized
            return sims
        else:
            # TF-IDF sparse cosine
            from sklearn.preprocessing import normalize
            base_i = clo_to_idx[base_clo_text]
            base_vec = clo_emb[base_i]
            other_idx = [clo_to_idx[t] for t in other_clo_texts]
            other_mat = clo_emb[other_idx]
            base_vec_n = normalize(base_vec)
            other_mat_n = normalize(other_mat)
            sims = (base_vec_n @ other_mat_n.T).A.reshape(-1)
            return sims

    # -----------------------------
    # 1) Course similarity (top 20 within same-level)
    # Define overall course similarity as:
    # For each CLO in base course, take max similarity to any CLO in other course; then average.
    # -----------------------------
    course_rows = []

    for base_course, base_clos in course_to_clos.items():
        lvl = df.loc[df["course"] == base_course, "level_key"].iloc[0]
        same_level_courses = [c for c in level_to_courses.get(lvl, []) if c != base_course]

        for other_course in same_level_courses:
            other_clos = course_to_clos.get(other_course, [])
            if not other_clos:
                continue

            # For each base CLO compute best match in other course
            bests = []
            for b in base_clos:
                sims = sim_one_to_many(b, other_clos)
                bests.append(float(np.max(sims)) if len(sims) else 0.0)

            overall = float(np.mean(bests)) if bests else 0.0

            course_rows.append({
                "base_course": base_course,
                "other_course": other_course,
                "overall": overall
            })

    df_course = pd.DataFrame(course_rows)
    df_course.sort_values(["base_course", "overall"], ascending=[True, False], inplace=True)
    df_course.drop_duplicates(subset=["base_course", "other_course"], keep="first", inplace=True)

    # Keep top 20 per base course
    df_course_top = df_course.groupby("base_course", group_keys=False).head(20).copy()
    df_course_top.to_csv(OUT_COURSE, index=False)
    print(f"Wrote {OUT_COURSE} ({len(df_course_top):,} rows)")

    # Build lookup for top courses per base_course
    top_courses_map = {
        bc: grp["other_course"].tolist()
        for bc, grp in df_course_top.groupby("base_course")
    }

    # -----------------------------
    # 2) CLO similarity TOP (top 20 across same-level courses)
    # -----------------------------
    clo_top_rows = []

    for base_course, base_clos in course_to_clos.items():
        lvl = df.loc[df["course"] == base_course, "level_key"].iloc[0]
        same_level_courses = [c for c in level_to_courses.get(lvl, []) if c != base_course]

        # Collect ALL CLOs from same-level courses
        pool_courses = same_level_courses
        pool_clos = []
        pool_course_labels = []
        for c in pool_courses:
            clos = course_to_clos.get(c, [])
            for clo in clos:
                pool_clos.append(clo)
                pool_course_labels.append(c)

        if not pool_clos:
            continue

        # For each base CLO, compute similarities to all pool CLOs
        for b in base_clos:
            sims = sim_one_to_many(b, pool_clos)

            # Get top 20 indices
            if len(sims) == 0:
                continue
            top_idx = np.argsort(-sims)[:20]

            seen_pairs = set()
            count = 0
            for idx in top_idx:
                cc = pool_course_labels[int(idx)]
                cclo = pool_clos[int(idx)]
                key = (cc, cclo)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                clo_top_rows.append({
                    "base_course": base_course,
                    "base_clo": b,
                    "compare_course": cc,
                    "compare_clo": cclo,
                    "similarity": float(sims[int(idx)])
                })
                count += 1
                if count >= 20:
                    break

    df_clo_top = pd.DataFrame(clo_top_rows)
    df_clo_top.sort_values(["base_course", "base_clo", "similarity"], ascending=[True, True, False], inplace=True)
    df_clo_top.drop_duplicates(
        subset=["base_course", "base_clo", "compare_course", "compare_clo"],
        keep="first",
        inplace=True
    )
    df_clo_top.to_csv(OUT_CLO_TOP, index=False)
    print(f"Wrote {OUT_CLO_TOP} ({len(df_clo_top):,} rows)")

    # -----------------------------
    # 3) DRILLDOWN FILE (THIS IS WHAT YOU NEED)
    # For each base_course/base_clo and each of its top-20 courses,
    # write TOP K CLOs INSIDE that clicked course.
    # -----------------------------
    K = 50  # you can change this (50 is nice for drilldown)

    drill_rows = []

    for base_course, base_clos in course_to_clos.items():
        top_courses = top_courses_map.get(base_course, [])
        if not top_courses:
            continue

        for b in base_clos:
            for cc in top_courses:
                other_clos = course_to_clos.get(cc, [])
                if not other_clos:
                    continue

                sims = sim_one_to_many(b, other_clos)
                if len(sims) == 0:
                    continue

                order = np.argsort(-sims)[:K]
                seen = set()
                n = 0
                for j in order:
                    clo_txt = other_clos[int(j)]
                    if clo_txt in seen:
                        continue
                    seen.add(clo_txt)

                    drill_rows.append({
                        "base_course": base_course,
                        "base_clo": b,
                        "compare_course": cc,
                        "compare_clo": clo_txt,
                        "similarity": float(sims[int(j)])
                    })
                    n += 1
                    if n >= K:
                        break

    df_drill = pd.DataFrame(drill_rows)
    df_drill.sort_values(
        ["base_course", "base_clo", "compare_course", "similarity"],
        ascending=[True, True, True, False],
        inplace=True
    )
    df_drill.drop_duplicates(
        subset=["base_course", "base_clo", "compare_course", "compare_clo"],
        keep="first",
        inplace=True
    )
    df_drill.to_csv(OUT_DRILL, index=False)
    print(f"Wrote {OUT_DRILL} ({len(df_drill):,} rows)")

    print("\nDONE.\nNow commit/push these CSVs and redeploy.")


if __name__ == "__main__":
    main()
