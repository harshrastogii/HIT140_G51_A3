# HIT140 Assessment 3: Bat vs Rat Foraging Analysis

## Group 51 - Team Members
- Harsh Rastogi - S386401
- Hena Akter - S383478
- Princyben Chetankumnar Patel - S388343
- Renish Rajeshkumar Vekariya - S374427

## Project Overview
This project analyzes foraging behavior of Egyptian Fruit Bats in the presence of Black Rats across different seasons.

**Research Questions:**
1. Do bats perceive rats as predators?
2. Do bat behaviors change seasonally?

## Repository Contents

### Data Files
- `dataset1.csv` - 907 bat landing observations
- `dataset2.csv` - 2,123 thirty-minute observation periods

### Code
- `analysis.py` - Complete Python analysis script
- `requirements.txt` - Required Python packages

### Output Files
- `plot1_vigilance_comparison.png`
- `plot2_risk_behavior.png`
- `plot3_seasonal_vigilance.png`
- `plot4_rat_activity_seasons.png`
- `plot5_risk_by_season.png`
- `plot6_correlation_matrix.png`
- `plot7_monthly_trends.png`
- `plot8_reward_success_analysis.png`

## Requirements

**Python Version:** 3.11+

**Required Libraries:**
```
pandas==2.2.3
numpy==2.1.3
matplotlib==3.9.2
seaborn==0.13.2
scipy==1.14.1
```

## How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Analysis
```bash
python3 analysis.py
```

### Expected Output
- Console displays statistical results for both investigations
- 8 PNG visualization files are generated in the same directory

## Key Findings

### Investigation A: Predator Perception
- NO significant difference in vigilance when rats present (p = 0.83)
- Risk-avoiders: 84.3% success vs Risk-takers: 21.8% success
- Evidence suggests competitive rather than predatory relationship

### Investigation B: Seasonal Changes
- Significant seasonal difference in vigilance (p = 0.006)
- Season 1 vigilance: 12.85 sec vs Season 0: 6.04 sec
- Rat activity 2.5x higher in Season 1
- Success rate higher in Season 1 (56.7% vs 36.4%)

## Data Source

Chen, X., Harten, L., Rachum, A., Attia, L., & Yovel, Y. (2025). Complex competition interactions between Egyptian fruit bats and black rats in the real world. Mendeley Data, V1. https://doi.org/10.17632/gt7j39b2cf.1

## License

Educational project for HIT140 Foundations of Data Science, Charles Darwin University, S2 2025.


## 📁 REPO STRUCTURE

```
HIT140_G51_A3/

├── README.md
├── requirements.txt
├── analysis.py
├── dataset1.csv
├── dataset2.csv
├── plot1_vigilance_comparison.png
├── plot2_risk_behavior.png
├── plot3_seasonal_vigilance.png
├── plot4_rat_activity_seasons.png
├── plot5_risk_by_season.png
├── plot6_correlation_matrix.png
├── plot7_monthly_trends.png
├── plot8_reward_success_analysis.png
└── Turnitin receipt_HIT140 G51 A3.docx.pdf
```

