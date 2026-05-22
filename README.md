# F1 Predictor with LLM Integration

## Overview
F1 Predictor is an AI-powered Formula 1 race prediction system that combines **Machine Learning** and **Large Language Models (LLMs)** to predict race outcomes and generate intelligent race analysis.

The project uses historical Formula 1 race data from the FastF1 API, processes sector and lap timings, predicts race performance using a Machine Learning model, and generates AI commentary using an LLM.

---

# Features

- Formula 1 race prediction using Machine Learning
- Driver performance analysis
- Sector-wise timing analysis
- AI-generated race commentary
- Natural language race insights
- CSV export support
- HuggingFace Transformer integration
- Optional OpenAI GPT integration

---

# Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- fastf1
- pandas
- numpy
- scikit-learn
- transformers
- torch

## Machine Learning
- GradientBoostingRegressor

## LLM / NLP
- HuggingFace Transformers
- DistilGPT2
- OpenAI GPT (Optional)

---

# Project Workflow

1. Collect historical F1 race data using FastF1
2. Extract lap times and sector times
3. Preprocess and clean the data
4. Train Machine Learning prediction model
5. Predict race performance
6. Generate AI-powered race analysis using LLMs
7. Export predictions and results

---

# Dataset Used

The project uses:
- 2024 Chinese Grand Prix race data
- Hypothetical 2025 qualifying data

Data includes:
- Lap Times
- Sector 1 Times
- Sector 2 Times
- Sector 3 Times
- Qualifying Times

---

# Machine Learning Model

The project uses:
- Gradient Boosting Regressor

### Input Features
- Qualifying Time
- Sector 1 Time
- Sector 2 Time
- Sector 3 Time

### Output
- Predicted Race Lap Time

---

# LLM Integration

The project integrates Large Language Models to generate:
- Race commentary
- Strategic insights
- Driver performance analysis
- Winner predictions
- AI-generated explanations

### Default LLM
- DistilGPT2 (HuggingFace)

### Optional
- OpenAI GPT-4o-mini
- GPT-4

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/f1-predictor-llm.git
cd f1-predictor-llm
```

---

# Install Dependencies

```bash
pip install fastf1 pandas numpy scikit-learn transformers torch openai
```

---

# Run the Project

```bash
python f1_predictor.py
```

---

# Output

The project produces:

- Predicted race rankings
- AI-generated race analysis
- Driver performance insights
- CSV export file

Example:

```text
🏁 Predicted 2025 Chinese GP Results 🏁

1. Max Verstappen
2. Lando Norris
3. Charles Leclerc
...
```

---

# CSV Export

Predictions are automatically saved as:

```text
f1_predictions_with_llm.csv
```

---

# Optional OpenAI Integration

To use OpenAI GPT models:

1. Install OpenAI package

```bash
pip install openai
```

2. Add your API key

```python
client = OpenAI(api_key="YOUR_API_KEY")
```

3. Uncomment the OpenAI section in the code.

---

# Skills Demonstrated

This project demonstrates:

- Machine Learning
- Data Science
- Sports Analytics
- Natural Language Processing
- LLM Integration
- Predictive Modeling
- AI Commentary Generation
- Python Development

---

# Future Improvements

Possible future enhancements:

- Real-time race prediction
- Weather integration
- Tire strategy analysis
- Deep Learning models (LSTM)
- Streamlit dashboard
- Live telemetry analysis
- Multi-race season simulation
- Team strategy prediction

---

# Project Structure

```text
F1-Predictor/
│
├── f1_predictor.py
├── f1_predictions_with_llm.csv
├── README.md
├── requirements.txt
└── f1_cache/
```

---

# Author

Developed as an AI + Machine Learning based Formula 1 analytics project.

---

# License

This project is for educational and research purposes.
