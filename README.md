# Formula 1 Predictor

A Machine Learning-Powered Formula 1 Race Predictor Using Real Lap Data, Qualifying Results, Weather Forecasts, And Team Performance Metrics.

## Features

- ✅ Loads And Processes F1 Session & Lap Data (FastF1)
- ✅ Integrates Real-Time Weather Forecasts (OpenWeatherMap)
- ✅ Predicts Race Results Using Gradient Boosting Regression
- ✅ Outputs Predicted Race Times & Podium Finishers
- ⏳ Add Support For More Circuits (Coming Soon)

---

## Getting Started

### Prerequisites

Ensure You Have The Following Installed:

- Python 3.8+
- pip

---

## Installation

1. Clone The Repository:

   ```bash
   git clone https://github.com/yourusername/formula1-predictor.git
   cd formula1-predictor
   ```

2. Install Dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set Up Environment Variables:
   - Create a `.env` file in the project root:
     ```
     OWM_API_KEY=your_openweathermap_api_key
     ```

---

## Running The Predictor

```bash
python prediction.py
```

The script will output predicted race times and the top 3 podium finishers for the next Monaco GP.

---

## Project Structure

```
formula1-predictor/
├── prediction.py      # Main Prediction Script
├── f1_cache/          # FastF1 Data Cache
├── .env               # API Keys (Not Tracked In Git)
├── requirements.txt   # Python Dependencies
└── README.md          # Project Documentation
```

---

## Technologies Used

- Python 3.8+
- FastF1
- scikit-learn
- pandas
- matplotlib
- OpenWeatherMap API
- dotenv

---