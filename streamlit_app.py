import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from streamlit_carousel_uui import uui_carousel

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(page_title="SmartEatAI", layout="wide")

FEATURES = ['calories', 'fat_content', 'carbohydrate_content', 'protein_content']
MACRO_WEIGHTS = np.array([1.5, 0.8, 1.0, 1.2])

DIET_LABELS = {
    "high_protein": "High Protein",
    "low_carb": "Low Carb",
    "vegan": "Vegan",
    "vegetarian": "Vegetarian",
    "low_calorie": "Low Calorie",
    "high_fiber": "High Fiber"
}

MEAL_COLORS = {
    "Breakfast": "#f39c12",
    "Lunch": "#2980b9",
    "Dinner": "#8e44ad",
    "Snack": "#16a085"
}

# --------------------------------------------------
# LOAD RESOURCES
# --------------------------------------------------
@st.cache_resource
def load_resources():
    df = load("files/df_recetas.joblib")
    scaler = load("files/scaler.joblib")
    knn = load("files/knn.joblib")

    X_scaled_all = scaler.transform(df[FEATURES])
    return df, scaler, knn, X_scaled_all

df_recetas, scaler, knn, X_scaled_all = load_resources()

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def render_tags(tags, color="#34495e"):
    html = ""
    for tag in tags:
        html += f"<span style='background:{color};color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{tag}</span>"
    st.markdown(html, unsafe_allow_html=True)

def render_diet_tags(diets):
    html = ""
    for d in diets:
        label = DIET_LABELS.get(d, d)
        html += f"<span style='background:#16a085;color:white;padding:4px 10px;border-radius:12px;margin-right:6px;font-size:13px'>{label}</span>"
    st.markdown(html, unsafe_allow_html=True)

def get_used_recipe_ids(exclude_id=None):
    if (
        "recipes" not in st.session_state
        or st.session_state.recipes is None
        or st.session_state.recipes.empty
        or "id" not in st.session_state.recipes.columns
    ):
        return set()

    ids = set(st.session_state.recipes["id"].dropna().tolist())

    if exclude_id is not None:
        ids.discard(exclude_id)

    return ids


def normalize_to_list(value):
    """
    Converts a value that may be:
    - list
    - comma-separated string
    - JSON string
    into a clean Python list of strings
    """
    if value is None:
        return []

    # Already a list
    if isinstance(value, list):
        return value

    # JSON list as string
    if isinstance(value, str) and value.strip().startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Comma-separated string
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]

    return []

import json

def safe_to_list(value):
    """
    Convierte distintos formatos a lista de strings:
    - Lista real -> lista
    - JSON string -> lista
    - String separado por comas -> lista
    - None / NaN -> []
    """
    if value is None:
        return []

    # Si ya es lista
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    # Si es string
    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # Intentar JSON
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass

        # Fallback: separado por comas
        return [v.strip() for v in value.split(",") if v.strip()]

    return []

# --------------------------------------------------
# RECOMMENDATION LOGIC
# --------------------------------------------------
def recommend_recipes(macros, diets, n, used_ids=None):
    if used_ids is None:
        used_ids = set()

    user_df = pd.DataFrame([macros], columns=FEATURES)
    user_scaled = scaler.transform(user_df) * MACRO_WEIGHTS
    X_weighted = X_scaled_all * MACRO_WEIGHTS

    if diets:
        mask = df_recetas["diet_type"].str.contains("|".join(diets), case=False, na=False)
        df_search = df_recetas[mask].copy()
        X_search = X_weighted[mask.values]
    else:
        df_search = df_recetas.copy()
        X_search = X_weighted

    # Quitar recetas ya usadas
    if used_ids:
        mask_used = ~df_search["id"].isin(used_ids)
        df_search = df_search[mask_used]
        X_search = X_search[mask_used.values]

    if df_search.empty:
        return pd.DataFrame()

    dist = np.linalg.norm(X_search - user_scaled, axis=1)
    df_search["dist"] = dist

    return df_search.sort_values("dist").head(n).reset_index(drop=True)

def swap_similar_unique(recipe_id, used_ids, max_candidates=30):
    idx_list = df_recetas.index[df_recetas["id"] == recipe_id].tolist()
    if not idx_list:
        return None

    idx = idx_list[0]

    vec_df = df_recetas.loc[[idx], FEATURES]
    vec = scaler.transform(vec_df) * MACRO_WEIGHTS

    _, indices = knn.kneighbors(vec, n_neighbors=max_candidates)

    for i in indices[0][1:]:
        candidate = df_recetas.iloc[i]
        if candidate["id"] not in used_ids:
            return candidate.copy()

    return None

# --------------------------------------------------
# USER MACROS
# --------------------------------------------------
def estimate_bodyfat(sex, category):
    mapping = {
        "Male": {"Lean": 12, "Normal": 18, "Stocky": 25, "Obese": 32},
        "Female": {"Lean": 20, "Normal": 26, "Stocky": 33, "Obese": 40}
    }
    return mapping[sex][category]

def calculate_macros(sex, age, height, weight, bf, activity, goal):
    lean_mass = weight * (1 - bf / 100)

    if sex == "Male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_factor = {
        "Sedentary": 1.2,
        "Light": 1.375,
        "Moderate": 1.55,
        "High": 1.725,
        "Very High": 1.9
    }[activity]

    tdee = bmr * activity_factor

    if goal == "Gain Muscle":
        calories = tdee * 1.1 + 150
        protein = lean_mass * 2.2
        diets = ["high_protein", "high_fiber"]
    elif goal == "Lose Weight":
        calories = tdee * 0.8
        protein = lean_mass * 2.2
        diets = ["low_carb", "low_calorie"]
    else:
        calories = tdee
        protein = lean_mass * 2.0
        diets = ["vegetarian", "high_fiber"]

    fat = (calories * 0.25) / 9
    carbs = (calories - (protein * 4 + fat * 9)) / 4

    return {
        "calories": round(calories),
        "protein": round(protein),
        "fat": round(fat),
        "carbs": round(carbs),
        "recommended_diets": diets
    }

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("""
<style>
button[kind="primary"] {
    background-color: #e74c3c !important;
    border-color: #e74c3c !important;
}
button[kind="primary"]:hover {
    background-color: #c0392b !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🥗 SmartEatAI")
st.caption("Intelligent meal recommendations based on your macros")

st.header("Profile Setup")

with st.form("user_form", border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        sex = st.selectbox("Sex", ["Male", "Female"])
        height = st.number_input("Height (cm)", 140, 220, 175)

    with c2:
        age = st.number_input("Age", 15, 90, 30)
        weight = st.number_input("Weight (kg)", 40, 200, 75)

    with c3:
        meals_per_day = st.number_input("Meals per day", 3, 6, 3)
        body = st.selectbox("Body Type", ["Lean", "Normal", "Stocky", "Obese"])

    activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "High", "Very High"])
    goal = st.selectbox("Main Goal", ["Gain Muscle", "Lose Weight", "Maintenance"])

    submit = st.form_submit_button("Generate Personalized Plan", use_container_width=True, type="primary")

# --------------------------------------------------
# PROCESS
# --------------------------------------------------
if submit:
    bf = estimate_bodyfat(sex, body)
    macros = calculate_macros(sex, age, height, weight, bf, activity, goal)
    st.session_state.macros = macros

# --------------------------------------------------
# DIET SELECTOR
# --------------------------------------------------
if "macros" in st.session_state:
    macros = st.session_state.macros
    recommended = macros["recommended_diets"]

    options = []
    for k, v in DIET_LABELS.items():
        if k in recommended:
            options.append(f"{v} [Recommended]")
        else:
            options.append(v)

    selected = st.multiselect(
        "Diet preferences",
        options,
        default=[o for o in options if "[Recommended]" in o]
    )

    def labels_to_keys(selected_labels):
        keys = []
        for s in selected_labels:
            clean = s.replace(" [Recommended]", "")
            for k, v in DIET_LABELS.items():
                if v == clean:
                    keys.append(k)
        return keys

    selected_diets = labels_to_keys(selected)

    st.session_state.selected_diets = selected_diets

    used_ids = get_used_recipe_ids()

    st.session_state.recipes = recommend_recipes(
        {
            "calories": macros["calories"] / meals_per_day,
            "fat_content": macros["fat"] / meals_per_day,
            "carbohydrate_content": macros["carbs"] / meals_per_day,
            "protein_content": macros["protein"] / meals_per_day,
        },
        selected_diets,
        meals_per_day,
        used_ids=used_ids
    )


# --------------------------------------------------
# MACROS SUMMARY CARDS
# --------------------------------------------------
if "macros" in st.session_state and "recipes" in st.session_state:
    st.subheader("📊 Daily Macro Progress")

    macros = st.session_state.macros
    recipes_df = st.session_state.recipes

    total_cal = recipes_df["calories"].sum()
    total_protein = recipes_df["protein_content"].sum()
    total_fat = recipes_df.get("fat_content", pd.Series(0)).sum()
    total_carb = recipes_df["carbohydrate_content"].sum()

    def macro_bar(label, value, total, color):
        pct = min(1.0, value / total) if total > 0 else 0
        html = f"""
        <div style="margin-bottom:10px">
            <b>{label}:</b> {value:.0f} / {total:.0f}
            <div style="background:#eee;width:100%;height:18px;border-radius:8px;overflow:hidden">
                <div style="width:{pct*100:.1f}%;height:100%;background:{color};"></div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        macro_bar("Calories", total_cal, macros["calories"], "#f39c12")
        macro_bar("Fat", total_fat, macros["fat"], "#27ae60")

    with col2:
        macro_bar("Protein", total_protein, macros["protein"], "#e74c3c")
        macro_bar("Carbohydrates", total_carb, macros["carbs"], "#2980b9")

# --------------------------------------------------
# DISPLAY RECIPES
# --------------------------------------------------
def card_container():
    return st.container(border=True)

# --------------------------------------------------
# DISPLAY RECIPES
# --------------------------------------------------
if "recipes" in st.session_state:
    st.header("🍽️ Recommended Meals")

    df_rec = st.session_state.recipes

    for idx, row in df_rec.iterrows():
        with st.container(border=True):

            # --- Title ---
            st.subheader(f"Meal {idx + 1}: {row['name']}")

            c1, c2 = st.columns([1, 2])

            # -----------
            # LEFT COLUMN
            # -----------
            with c1:
                imgs = row["images"].split(", ")
                slides = [
                    {"image": url, "title": "", "description": ""}
                    for url in imgs[:3]
                ]

                uui_carousel(
                    items=slides,
                    variant="sm",
                    key=f"carousel_{row['id']}_{idx}"  # unique key
                )

            # ------------
            # RIGHT COLUMN
            # ------------
            with c2:
                # --- Meal type tags ---
                meal_types = row.get("meal_type", [])
                if isinstance(meal_types, str):
                    meal_types = json.loads(meal_types)

                tags_html = ""
                for mt in meal_types:
                    color = MEAL_COLORS.get(mt, "#34495e")
                    tags_html += (
                        f"<span style='background:{color};color:white;"
                        f"padding:4px 10px;border-radius:12px;"
                        f"margin-right:6px;font-size:13px'>{mt}</span>"
                    )
                st.markdown(tags_html, unsafe_allow_html=True)

                # --- Diet tags ---
                diet_types = normalize_to_list(row.get("diet_type"))
                render_diet_tags(diet_types)


                st.write(
                    f"**🔥 Calories:** {row['calories']} kcal  \n"
                    f"**🥩 Protein:** {row['protein_content']} g | "
                    f"**🥑 Fat:** {row.get('fat_content', 0)} g | "
                    f"**🍞 Carbs:** {row['carbohydrate_content']} g"
                )

                # --- Ingredients ---
                st.write("**Ingredients:**")
                ingredients = safe_to_list(row.get("recipe_ingredient_parts"))

                #for ing in ingredients:
                #    st.write(f"• {ing}")

                # -----------
                # SWAP BUTTON
                # -----------
                if st.button("🔄 Swap for similar", key=f"btn_swp_{row['id']}_{idx}"):
                    used_ids = get_used_recipe_ids(exclude_id=row["id"])
                    nueva = swap_similar_unique(row["id"], used_ids)

                    if nueva is not None:
                        df_new = st.session_state.recipes.copy().reset_index(drop=True)

                        for col in df_new.columns:
                            if col not in nueva:
                                nueva[col] = df_new.loc[idx, col]

                        df_new.loc[idx] = nueva[df_new.columns]
                        st.session_state.recipes = df_new

                        st.success("New recipe available ✨")
                        st.rerun()
                    else:
                        st.warning("No alternative recipes 😕")


