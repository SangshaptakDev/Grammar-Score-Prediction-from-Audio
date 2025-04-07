# 🎯 SHL AI Internship Assessment: Grammar Scoring from Audio Features

This project was built as part of the **SHL AI Internship Hiring Assessment 2025**, where the goal is to **predict grammar proficiency scores** based on spoken audio samples.

> 🧠 The solution leverages extracted audio and grammar-based features (no deep speech models or transcriptions) and applies classical machine learning techniques for regression-based scoring.

---

## 📌 Problem Overview

- Predict a speaker’s **grammar score** (numerical) from their **spoken audio**.
- Audio files were preprocessed to extract key features.
- A basic **ensemble of regression models** was used to make final predictions.

---

#### 🧪 Approach Summary

### 🧾 Feature Set

- Extracted audio features (MFCCs, pitch, etc.) using `librosa`.
- Grammar-check-based features (number of grammar errors, sentence structure stats, etc.) using `language_tool_python`.

### 🤖 Models Used

- `RandomForestRegressor`
- `XGBRegressor`
- `Ridge Regression`
- Combined using **Voting Regressor** (unweighted average)

 
### 🧠 Why This Approach?
This project avoids heavy ASR (Automatic Speech Recognition) or deep learning pipelines for faster experimentation and explainability. The idea was to:

Capture grammar proficiency through measurable linguistic and acoustic cues.

Keep the pipeline lightweight, interpretable, and reproducible.

Lay a strong baseline for potential improvement using advanced embeddings (like Wav2Vec2) or LLM-based scoring.

🧮 Evaluation Metrics
To assess prediction quality:

MAE (Mean Absolute Error): Measures average error in predicted scores.

RMSE (Root Mean Square Error): Penalizes larger errors more strongly.

R² Score: Measures how well the model explains the variance in the target.  
