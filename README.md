# EPL Match Predictor ⚽️

A Machine Learning powered web application that predicts the outcomes of English Premier League matches. This project integrates a Random Forest classification model with a Flask web interface to provide real-time win/draw/loss probabilities based on historical match data, team form, and dynamic ELO ratings.

## 📖 Project Overview

The goal of this project was to build a full-stack data science application that moves beyond simple static analysis. By engineering dynamic features—specifically **Rolling Averages** to capture recent team form and **ELO Ratings** to capture long-term team strength—the model aims to predict match results with higher nuance than simple table standings.

**Key Features:**

* **Machine Learning Pipeline:** End-to-end data processing, feature extraction, and model training.
* **Dynamic Feature Engineering:** Custom implementation of ELO ratings and rolling temporal statistics.
* **Interactive Web Interface:** A clean, responsive frontend built with HTML/CSS and served via Flask.
* **Probabilistic Output:** Returns the probability confidence for a Win, Draw, or Loss.

## 🛠 Tech Stack

* **Language:** Python 3.12
* **Web Framework:** Flask
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Manipulation:** Pandas, NumPy
* **Frontend:** HTML5, CSS3, Jinja2 Templating

## ⚙️ How It Works (Methodology)

The core logic resides in `Model.py`, which processes raw match statistics (`match_data.csv`) through several stages before training.

### 1. Data Cleaning & Preprocessing

Raw match data is cleaned to standardise team names and remove non-predictive metadata (Attendance, Notes, Captains). Categorical variables such as `Venue` and `Team Names` are encoded using `LabelEncoder`.

### 2. Feature Engineering

To improve predictive accuracy, I engineered two specific types of features:

* **ELO Rating System:**
I implemented a custom ELO calculation engine from scratch.
* Every team starts with a baseline rating of 1500.
* Ratings are updated iteratively after every match row based on the result and the strength of the opponent (k=32).
* This allows the model to understand that beating a strong team is weightier than beating a weak one.


* **Rolling Statistics (Form):**
To capture "current form," the model calculates rolling averages for the last 3 matches for every team. Metrics include:
* Goals For/Against (GF, GA)
* Expected Goals (xG)
* Possession, Shots, and Shots on Target



### 3. Model Training

* **Algorithm:** Random Forest Classifier (`n_estimators=100`).
* **Validation:** Uses an 80/20 train-test split to evaluate accuracy.
* **Output:** The model predicts the outcome (Win/Draw/Loss) and provides probability distributions using `predict_proba`.

## 📂 Project Structure

```bash
.
├── Model.py              # ML Pipeline: Data cleaning, ELO calc, training, and prediction logic
├── app.py                # Flask application entry point and route handling
├── match_data.csv        # Historical EPL match data
├── templates/            # Jinja2 HTML templates
│   ├── base.html         # Layout skeleton
│   └── index.html        # Main interface
├── static/               # CSS and Assets
│   ├── stylesheets/
│   │   ├── styles.css    # Custom styling
│   │   └── reset.css     # CSS reset for consistency
│   └── assets/           # Images
└── README.md

```

## 🚀 Getting Started

### Prerequisites

* Python 3.x
* pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/epl-predictor.git
cd epl-predictor

```


2. **Install dependencies**
```bash
pip install flask pandas scikit-learn

```


3. **Run the application**
```bash
python app.py

```


4. **Access the App**
Open your browser and navigate to `http://127.0.0.1:5000/`.

## 📸 Screenshots
!web_interface_prediction


