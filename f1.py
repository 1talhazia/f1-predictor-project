# F1 Predictor – LLM Integrated Version

# ===============================
# F1 PREDICTOR WITH LLM INTEGRATION
# ===============================
# Features Added:
# 1. Machine Learning prediction using Gradient Boosting
# 2. LLM-generated race analysis
# 3. Natural language explanation of predictions
# 4. AI-powered race insights
# 5. Driver performance summaries
#
# Required Libraries:
# pip install fastf1 pandas numpy scikit-learn transformers torch openai
#
# Optional:
# If using OpenAI API:
# pip install openai
#
# =====================================

import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from transformers import pipeline
import warnings

warnings.filterwarnings("ignore")

# =====================================
# ENABLE FASTF1 CACHE
# =====================================
fastf1.Cache.enable_cache("./f1_cache")

print("\n==============================")
print("🏎️  F1 PREDICTOR WITH LLM AI")
print("==============================\n")

# =====================================
# LOAD RACE DATA
# =====================================
print("Loading 2024 Chinese GP data...")

session_2024 = fastf1.get_session(2024, "China", "R")
session_2024.load()

# =====================================
# EXTRACT LAP DATA
# =====================================
laps_2024 = session_2024.laps[[
    "Driver",
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time"
]].copy()

laps_2024.dropna(inplace=True)

# =====================================
# CONVERT TIMES TO SECONDS
# =====================================
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# =====================================
# PROCESS SECTOR TIMES
# =====================================
print("Processing sector times...")

sector_times_2024 = laps_2024.groupby("Driver")[[
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)"
]].mean().reset_index()

# =====================================
# HYPOTHETICAL 2025 QUALIFYING DATA
# =====================================
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Oscar Piastri",
        "George Russell",
        "Lando Norris",
        "Max Verstappen",
        "Lewis Hamilton",
        "Charles Leclerc",
        "Yuki Tsunoda",
        "Alexander Albon",
        "Esteban Ocon",
        "Nico Hülkenberg",
        "Fernando Alonso",
        "Lance Stroll",
        "Carlos Sainz Jr.",
        "Pierre Gasly"
    ],

    "QualifyingTime (s)": [
        90.641,
        90.723,
        90.793,
        90.817,
        90.927,
        91.021,
        91.638,
        91.706,
        91.625,
        91.632,
        91.688,
        91.773,
        91.840,
        91.992
    ]
})

# =====================================
# DRIVER CODE MAPPING
# =====================================
driver_mapping = {
    "Oscar Piastri": "PIA",
    "George Russell": "RUS",
    "Lando Norris": "NOR",
    "Max Verstappen": "VER",
    "Lewis Hamilton": "HAM",
    "Charles Leclerc": "LEC",
    "Yuki Tsunoda": "TSU",
    "Alexander Albon": "ALB",
    "Esteban Ocon": "OCO",
    "Nico Hülkenberg": "HUL",
    "Fernando Alonso": "ALO",
    "Lance Stroll": "STR",
    "Carlos Sainz Jr.": "SAI",
    "Pierre Gasly": "GAS"
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(driver_mapping)

# =====================================
# MERGE DATA
# =====================================
merged_data = qualifying_2025.merge(
    sector_times_2024,
    left_on="DriverCode",
    right_on="Driver",
    how="left"
)

# =====================================
# PREPARE FEATURES
# =====================================
X = merged_data[[
    "QualifyingTime (s)",
    "Sector1Time (s)",
    "Sector2Time (s)",
    "Sector3Time (s)"
]].fillna(0)

# =====================================
# TARGET VARIABLE
# =====================================
y = merged_data.merge(
    laps_2024.groupby("Driver")["LapTime (s)"].mean(),
    left_on="DriverCode",
    right_index=True
)["LapTime (s)"]

# =====================================
# TRAIN ML MODEL
# =====================================
print("Training Gradient Boosting model...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=38
)

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=38
)

model.fit(X_train, y_train)

# =====================================
# PREDICT RACE TIMES
# =====================================
predicted_race_times = model.predict(X)

qualifying_2025["PredictedRaceTime (s)"] = predicted_race_times

# =====================================
# RANK DRIVERS
# =====================================
qualifying_2025 = qualifying_2025.sort_values(
    by="PredictedRaceTime (s)"
)

# =====================================
# MODEL EVALUATION
# =====================================
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# =====================================
# PRINT RESULTS
# =====================================
print("\n🏁 Predicted 2025 Chinese GP Results 🏁")
print("(Based on 2024 performance data and hypothetical 2025 qualifying)\n")

print(
    qualifying_2025[["Driver", "PredictedRaceTime (s)"]]
    .to_string(index=False)
)

print(f"\n📊 Model Mean Absolute Error: {mae:.3f} seconds")

# =====================================
# =====================================
# LLM INTEGRATION SECTION
# =====================================
# =====================================

print("\nLoading LLM model for AI race analysis...")

# =====================================
# LOAD LLM
# =====================================
# Using HuggingFace text generation pipeline
# Lightweight model for local execution
# =====================================

try:
    generator = pipeline(
        "text-generation",
        model="distilgpt2"
    )

    llm_available = True

except Exception as e:
    print("LLM could not be loaded:")
    print(e)
    llm_available = False

# =====================================
# GENERATE DRIVER SUMMARY DATA
# =====================================
summary_text = ""

for index, row in qualifying_2025.iterrows():
    summary_text += (
        f"Driver: {row['Driver']}, "
        f"Predicted Race Time: {row['PredictedRaceTime (s)']:.2f} seconds.\n"
    )


# =====================================
# GENERATE AI ANALYSIS
# =====================================
if llm_available:

    print("\n🤖 Generating AI-powered race analysis...\n")

    try:
        ai_analysis = generator(
            llm_prompt,
            max_length=350,
            num_return_sequences=1,
            temperature=0.8,
            truncation=True
        )

        generated_text = ai_analysis[0]["generated_text"]

        print("=" * 60)
        print("🧠 LLM RACE ANALYSIS")
        print("=" * 60)
        print(generated_text)
        print("=" * 60)

    except Exception as e:
        print("Error during AI text generation:")
        print(e)

# =====================================
# DRIVER PERFORMANCE INSIGHTS
# =====================================
print("\n📈 DRIVER PERFORMANCE INSIGHTS\n")

best_driver = qualifying_2025.iloc[0]
worst_driver = qualifying_2025.iloc[-1]

print(f"🏆 Expected Winner: {best_driver['Driver']}")
print(f"⏱️ Predicted Time: {best_driver['PredictedRaceTime (s)']:.2f} seconds\n")

print(f"📉 Slowest Predicted Driver: {worst_driver['Driver']}")
print(f"⏱️ Predicted Time: {worst_driver['PredictedRaceTime (s)']:.2f} seconds\n")

# =====================================
# TOP 5 DRIVERS
# =====================================
print("🔥 Top 5 Predicted Drivers:\n")

for i in range(min(5, len(qualifying_2025))):

    row = qualifying_2025.iloc[i]

    print(
        f"P{i+1}: {row['Driver']} --> "
        f"{row['PredictedRaceTime (s)']:.2f} seconds"
    )

# =====================================
# SAVE RESULTS TO CSV
# =====================================
qualifying_2025.to_csv(
    "f1_predictions_with_llm.csv",
    index=False
)

print("\n✅ Predictions saved to 'f1_predictions_with_llm.csv'")


# =====================================
# END OF PROGRAM
# =====================================
print("\n==============================")
print("✅ F1 Predictor Execution Complete")
print("==============================")
