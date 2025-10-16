"""
HIT140 Assessment 3: Bat vs Rat Analysis
Investigation A: Do bats perceive rats as predators?
Investigation B: Seasonal changes in bat behavior

Team Members: [Add your names here]
Date: October 2025
"""

# ===== STEP 1: IMPORT LIBRARIES =====
print("Loading libraries...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("Libraries loaded successfully!")

# ===== STEP 2: LOAD DATASETS =====
print("\nLoading datasets...")
df1 = pd.read_csv('dataset1.csv')
df2 = pd.read_csv('dataset2.csv')
print(f"Dataset 1 loaded: {df1.shape[0]} rows, {df1.shape[1]} columns")
print(f"Dataset 2 loaded: {df2.shape[0]} rows, {df2.shape[1]} columns")

# ===== STEP 3: EXPLORE THE DATA =====
print("\n" + "="*50)
print("DATASET 1 - BAT LANDINGS")
print("="*50)
print("\nFirst few rows:")
print(df1.head())
print("\nColumn names:")
print(df1.columns.tolist())
print("\nData types:")
print(df1.dtypes)
print("\nMissing values:")
print(df1.isnull().sum())
print("\nBasic statistics:")
print(df1.describe())

print("\n" + "="*50)
print("DATASET 2 - RAT ARRIVALS")
print("="*50)
print("\nFirst few rows:")
print(df2.head())
print("\nColumn names:")
print(df2.columns.tolist())
print("\nData types:")
print(df2.dtypes)
print("\nMissing values:")
print(df2.isnull().sum())
print("\nBasic statistics:")
print(df2.describe())

# ===== STEP 4: DATA CLEANING =====
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Dataset 1 Cleaning
print("\nCleaning Dataset 1...")

# Convert time columns to datetime (suppress the warning by specifying dayfirst)
print("Converting time columns to datetime format...")
df1['start_time'] = pd.to_datetime(df1['start_time'], dayfirst=True, errors='coerce')
df1['rat_period_start'] = pd.to_datetime(df1['rat_period_start'], dayfirst=True, errors='coerce')
df1['rat_period_end'] = pd.to_datetime(df1['rat_period_end'], dayfirst=True, errors='coerce')
df1['sunset_time'] = pd.to_datetime(df1['sunset_time'], dayfirst=True, errors='coerce')

# Handle missing values in numerical columns
print("Handling missing values...")
missing_cols = df1.columns[df1.isnull().any()].tolist()
print(f"Columns with missing values: {missing_cols}")

# Fill missing bat_landing_to_food with median (proper way without warning)
if 'bat_landing_to_food' in df1.columns:
    median_value = df1['bat_landing_to_food'].median()
    df1['bat_landing_to_food'] = df1['bat_landing_to_food'].fillna(median_value)
    print(f"Filled missing bat_landing_to_food with median: {median_value}")

# Remove rows where critical information is missing
initial_rows = len(df1)
df1 = df1.dropna(subset=['risk', 'reward', 'season'])
print(f"Removed {initial_rows - len(df1)} rows with missing critical data")

# Dataset 2 Cleaning
print("\nCleaning Dataset 2...")
df2['time'] = pd.to_datetime(df2['time'], dayfirst=True, errors='coerce')

# Check for negative values
print("Checking for invalid values...")
if (df2['rat_arrival_number'] < 0).any():
    print("Found negative rat arrivals - replacing with 0")
    df2.loc[df2['rat_arrival_number'] < 0, 'rat_arrival_number'] = 0

print("\nData cleaning complete!")
print(f"Final Dataset 1 size: {df1.shape[0]} rows")
print(f"Final Dataset 2 size: {df2.shape[0]} rows")

# ===== INVESTIGATION A: PREDATOR PERCEPTION =====
print("\n" + "="*50)
print("INVESTIGATION A: PREDATOR PERCEPTION ANALYSIS")
print("="*50)

# Create rats_present indicator
df1['rats_present'] = df1['seconds_after_rat_arrival'] > 0

# Analysis 1: Vigilance Behavior
print("\n--- Analysis 1: Bat Vigilance (Time to Approach Food) ---")
print(f"Average time before approaching food: {df1['bat_landing_to_food'].mean():.2f} seconds")
print(f"Median time before approaching food: {df1['bat_landing_to_food'].median():.2f} seconds")
print(f"Standard deviation: {df1['bat_landing_to_food'].std():.2f} seconds")

vigilance_with_rats = df1[df1['rats_present'] == True]['bat_landing_to_food']
vigilance_without_rats = df1[df1['rats_present'] == False]['bat_landing_to_food']

print(f"\nVigilance WITH rats present: {vigilance_with_rats.mean():.2f} seconds")
print(f"Vigilance WITHOUT rats: {vigilance_without_rats.mean():.2f} seconds")

# Statistical test
t_stat, p_value = stats.ttest_ind(vigilance_with_rats.dropna(), 
                                    vigilance_without_rats.dropna())
print(f"\nT-test results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: SIGNIFICANT difference (p < 0.05)")
else:
    print("Result: NO significant difference (p >= 0.05)")

# Analysis 2: Risk-Taking Behavior
print("\n--- Analysis 2: Risk-Taking Behavior ---")
risk_counts = df1['risk'].value_counts()
print("Risk behavior distribution:")
print(f"Risk-avoidance (0): {risk_counts.get(0, 0)} events ({risk_counts.get(0, 0)/len(df1)*100:.1f}%)")
print(f"Risk-taking (1): {risk_counts.get(1, 0)} events ({risk_counts.get(1, 0)/len(df1)*100:.1f}%)")

# Analysis 3: Reward Success Rate
print("\n--- Analysis 3: Reward Success Rate ---")
risk_takers = df1[df1['risk'] == 1]
risk_avoiders = df1[df1['risk'] == 0]

risk_taker_success = risk_takers['reward'].mean() * 100
risk_avoider_success = risk_avoiders['reward'].mean() * 100

print(f"Success rate for risk-takers: {risk_taker_success:.1f}%")
print(f"Success rate for risk-avoiders: {risk_avoider_success:.1f}%")

# Chi-square test
contingency_table = pd.crosstab(df1['risk'], df1['reward'])
chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value_chi:.4f}")

# Analysis 4: Timing
print("\n--- Analysis 4: Timing Relative to Rats ---")
print(f"Average time after rat arrival: {df1['seconds_after_rat_arrival'].mean():.2f} seconds")
print(f"Do bats land quickly or wait? Median: {df1['seconds_after_rat_arrival'].median():.2f} seconds")

print("\nInvestigation A Analysis Complete!")

# ===== INVESTIGATION B: SEASONAL CHANGES =====
print("\n" + "="*50)
print("INVESTIGATION B: SEASONAL ANALYSIS")
print("="*50)

# Check seasons in Dataset 1
print("\n--- Dataset 1: Seasons Available ---")
print(df1['season'].value_counts())
print("\nSeason labels in Dataset 1:")
print(f"Season 0: {df1[df1['season']==0].shape[0]} observations")
print(f"Season 1: {df1[df1['season']==1].shape[0]} observations")

# Create season mapping for Dataset 2 based on month
print("\n--- Creating Season Labels for Dataset 2 ---")
# Map months to seasons based on Dataset 1's pattern
month_to_season = df1.groupby('month')['season'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
print("Month to Season mapping:")
print(month_to_season)

# Apply mapping to Dataset 2
df2['season'] = df2['month'].map(month_to_season)
print("\nDataset 2 now has season labels:")
print(df2['season'].value_counts())

# Analysis 1: Vigilance by Season
print("\n--- Analysis 1: Vigilance Across Seasons ---")
vigilance_by_season = df1.groupby('season')['bat_landing_to_food'].agg(['mean', 'median', 'std', 'count'])
print(vigilance_by_season)

# T-test for two seasons
season_groups = [group['bat_landing_to_food'].dropna() for name, group in df1.groupby('season')]
if len(season_groups) == 2:
    t_stat_season, p_val_season = stats.ttest_ind(season_groups[0], season_groups[1])
    print(f"\nT-test comparing seasons:")
    print(f"Season 0 mean: {season_groups[0].mean():.2f} seconds")
    print(f"Season 1 mean: {season_groups[1].mean():.2f} seconds")
    print(f"T-statistic: {t_stat_season:.4f}")
    print(f"P-value: {p_val_season:.4f}")
    if p_val_season < 0.05:
        print("Result: SIGNIFICANT difference (p < 0.05)")
    else:
        print("Result: NO significant difference (p >= 0.05)")

# Analysis 2: Risk Behavior by Season
print("\n--- Analysis 2: Risk Behavior Across Seasons ---")
risk_by_season = pd.crosstab(df1['season'], df1['risk'], normalize='index') * 100
print("\nPercentage of risk-taking behavior by season:")
print(risk_by_season)

# Analysis 3: Rat Activity by Season
print("\n--- Analysis 3: Rat Activity Across Seasons (Dataset 2) ---")
rat_activity = df2.groupby('season')['rat_arrival_number'].agg(['mean', 'median', 'sum', 'std'])
print(rat_activity)

# Analysis 4: Bat Activity by Season
print("\n--- Analysis 4: Bat Landing Frequency Across Seasons (Dataset 2) ---")
bat_activity = df2.groupby('season')['bat_landing_number'].agg(['mean', 'median', 'sum', 'std'])
print(bat_activity)

# Analysis 5: Food Availability
print("\n--- Analysis 5: Food Availability Across Seasons ---")
food_by_season = df2.groupby('season')['food_availability'].agg(['mean', 'median', 'std'])
print(food_by_season)

# Analysis 6: Reward Success by Season
print("\n--- Analysis 6: Reward Success Across Seasons ---")
reward_by_season = df1.groupby('season')['reward'].agg(['mean', 'sum', 'count'])
reward_by_season['success_rate'] = reward_by_season['mean'] * 100
print(reward_by_season)

print("\nInvestigation B Analysis Complete!")

# ===== STEP 5: VISUALIZATIONS =====
print("\n" + "="*50)
print("CREATING VISUALIZATIONS")
print("="*50)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Plot 1: Vigilance Comparison
print("\nCreating Plot 1: Vigilance Comparison...")
plt.figure(figsize=(10, 6))
data_to_plot = [vigilance_with_rats.dropna(), vigilance_without_rats.dropna()]
plt.boxplot(data_to_plot, tick_labels=['Rats Present', 'No Rats'])
plt.ylabel('Time to Approach Food (seconds)', fontsize=12)
plt.xlabel('Condition', fontsize=12)
plt.title('Bat Vigilance: Rats Present vs Absent', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.savefig('plot1_vigilance_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: plot1_vigilance_comparison.png")
plt.close()

# Plot 2: Risk Behavior
print("Creating Plot 2: Risk Behavior...")
plt.figure(figsize=(8, 6))
risk_counts.plot(kind='bar', color=['green', 'red'])
plt.xlabel('Risk Behavior', fontsize=12)
plt.ylabel('Number of Events', fontsize=12)
plt.title('Distribution of Risk-Taking Behavior', fontsize=14, fontweight='bold')
plt.xticks([0, 1], ['Risk-Avoidance', 'Risk-Taking'], rotation=0)
plt.tight_layout()
plt.savefig('plot2_risk_behavior.png', dpi=300, bbox_inches='tight')
print("Saved: plot2_risk_behavior.png")
plt.close()

# Plot 3: Seasonal Vigilance
print("Creating Plot 3: Seasonal Vigilance...")
plt.figure(figsize=(10, 6))
df1['season_label'] = df1['season'].map({0: 'Season 0', 1: 'Season 1'})
df1.boxplot(column='bat_landing_to_food', by='season_label', figsize=(10, 6))
plt.ylabel('Time to Approach Food (seconds)', fontsize=12)
plt.xlabel('Season', fontsize=12)
plt.title('Bat Vigilance Across Seasons', fontsize=14, fontweight='bold')
plt.suptitle('')
plt.savefig('plot3_seasonal_vigilance.png', dpi=300, bbox_inches='tight')
print("Saved: plot3_seasonal_vigilance.png")
plt.close()

# Plot 4: Rat Activity by Season
print("Creating Plot 4: Rat Activity by Season...")
plt.figure(figsize=(10, 6))
rat_activity_means = df2.groupby('season')['rat_arrival_number'].mean()
seasons = ['Season 0', 'Season 1']
plt.bar(seasons, rat_activity_means.values, color='brown', alpha=0.7)
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Rat Arrivals per 30 min', fontsize=12)
plt.title('Rat Activity Across Seasons', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot4_rat_activity_seasons.png', dpi=300, bbox_inches='tight')
print("Saved: plot4_rat_activity_seasons.png")
plt.close()

# Plot 5: Risk by Season
print("Creating Plot 5: Risk Behavior by Season...")
plt.figure(figsize=(10, 6))
risk_by_season.plot(kind='bar', stacked=False, color=['green', 'red'])
plt.xlabel('Season', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Risk Behavior Distribution Across Seasons', fontsize=14, fontweight='bold')
plt.legend(['Risk-Avoidance', 'Risk-Taking'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot5_risk_by_season.png', dpi=300, bbox_inches='tight')
print("Saved: plot5_risk_by_season.png")
plt.close()

# Plot 6: Correlation Matrix
print("Creating Plot 6: Correlation Matrix...")
plt.figure(figsize=(10, 8))
numerical_cols = ['bat_landing_to_food', 'seconds_after_rat_arrival', 'risk', 
                  'reward', 'month', 'hours_after_sunset', 'season']
correlation_matrix = df1[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1)
plt.title('Correlation Matrix of Bat Behavior Variables', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('plot6_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: plot6_correlation_matrix.png")
plt.close()

# Plot 7: Monthly Trends
print("Creating Plot 7: Monthly Trends...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

monthly_vigilance = df1.groupby('month')['bat_landing_to_food'].mean()
ax1.plot(monthly_vigilance.index, monthly_vigilance.values, marker='o', linewidth=2, markersize=8)
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Average Vigilance (seconds)', fontsize=12)
ax1.set_title('Average Bat Vigilance by Month', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

monthly_rats = df2.groupby('month')['rat_arrival_number'].mean()
ax2.plot(monthly_rats.index, monthly_rats.values, marker='s', linewidth=2, markersize=8, color='brown')
ax2.set_xlabel('Month', fontsize=12)
ax2.set_ylabel('Average Rat Arrivals', fontsize=12)
ax2.set_title('Average Rat Arrivals by Month', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('plot7_monthly_trends.png', dpi=300, bbox_inches='tight')
print("Saved: plot7_monthly_trends.png")
plt.close()

# Plot 8: Reward Success Analysis
print("Creating Plot 8: Reward Success Analysis...")
plt.figure(figsize=(10, 6))
reward_analysis = df1.groupby(['season', 'risk'])['reward'].mean() * 100
reward_analysis = reward_analysis.unstack()
reward_analysis.plot(kind='bar', color=['green', 'red'])
plt.xlabel('Season', fontsize=12)
plt.ylabel('Success Rate (%)', fontsize=12)
plt.title('Reward Success Rate by Season and Risk Behavior', fontsize=14, fontweight='bold')
plt.legend(['Risk-Avoidance', 'Risk-Taking'])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('plot8_reward_success_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: plot8_reward_success_analysis.png")
plt.close()

print("\nAll visualizations created successfully!")
print("Check your folder for PNG files.")
print("\nAnalysis complete!")