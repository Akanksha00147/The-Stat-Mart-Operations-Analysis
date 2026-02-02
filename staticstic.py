# ==============================================================================
# IMPORT LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# ==============================================================================
# MODULE 1 & 2: INTRODUCTION, COLLECTION & ORGANIZATION
# Topics: Data types (Nominal, Ordinal, Ratio), Frequency Distribution
# ==============================================================================
print("--- MODULE 1 & 2: DATA GENERATION & ORGANIZATION ---")

# Setting a seed ensures the random numbers are the same every time you run this.
np.random.seed(42)
n = 1000  # Population size

# Simulating a dataset with various Data Types
data = {
    # Nominal Data: Names/Labels with no order
    'Order_ID': [f'ORD-{i}' for i in range(1, n+1)],
    'Warehouse': np.random.choice(['North', 'South', 'East', 'West'], n),
    'Category': np.random.choice(['Furniture', 'Tech', 'Clothing'], n, p=[0.2, 0.3, 0.5]),
    
    # Ordinal Data: Categories with a clear rank/order (1 to 5 stars)
    'Customer_Rating': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
    
    # Ratio Data: Numerical data with a true zero (Money, Time)
    'Delivery_Days': np.random.randint(1, 15, n),
    'Order_Value': np.random.normal(2000, 500, n).round(2), # Normal distribution around 2000
    'Discount_Pct': np.random.uniform(0, 30, n).round(1),
    
    # Qualitative/Binary Data
    'Returned': np.random.choice([0, 1], n, p=[0.85, 0.15]) # 0=No, 1=Yes
}

df = pd.DataFrame(data)

# Introducing correlations/patterns for later analysis:
# 1. Higher Discount -> Slightly Higher Order Value (Positive Correlation)
df['Order_Value'] += df['Discount_Pct'] * 15
# 2. 'North' Warehouse is slower (for Hypothesis Testing)
df.loc[df['Warehouse'] == 'North', 'Delivery_Days'] += 3

print("Data Preview (First 5 rows):")
print(df.head())
print("\nData Info (Types of Data):")
print(df.info())
print("-" * 60)

# ==============================================================================
# MODULE 3: DATA VISUALIZATION
# Topics: Bar charts, Histograms, Boxplots, Scatter plots
# ==============================================================================
print("\n--- MODULE 3: DATA VISUALIZATION ---")
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# 1. Bar Chart: Frequency of Orders by Category
plt.subplot(2, 2, 1)
sns.countplot(x='Category', data=df, palette='viridis')
plt.title('Bar Chart: Orders by Category')

# 2. Histogram: Distribution of Order Value
plt.subplot(2, 2, 2)
sns.histplot(df['Order_Value'], kde=True, color='teal')
plt.title('Histogram: Order Value Distribution')

# 3. Boxplot: Delivery Days by Warehouse (Detecting outliers/spread)
plt.subplot(2, 2, 3)
sns.boxplot(x='Warehouse', y='Delivery_Days', data=df, palette='coolwarm')
plt.title('Boxplot: Delivery Days per Warehouse')

# 4. Scatter Plot: Discount vs. Order Value (Visualizing relationship)
plt.subplot(2, 2, 4)
sns.scatterplot(x='Discount_Pct', y='Order_Value', data=df, alpha=0.6)
plt.title('Scatter Plot: Discount vs Order Value')

plt.tight_layout()
plt.show()
print("Graphs generated: Bar, Histogram, Boxplot, Scatter.")
print("-" * 60)

# ==============================================================================
# MODULE 4: MEASURES OF CENTRAL TENDENCY
# Topics: Mean, Median, Mode, Weighted Mean
# ==============================================================================
print("\n--- MODULE 4: CENTRAL TENDENCY ---")

# Mean (Average) Order Value
mean_val = df['Order_Value'].mean()

# Median (Middle) Delivery Days
median_days = df['Delivery_Days'].median()

# Mode (Most Frequent) Customer Rating
mode_rating = df['Customer_Rating'].mode()[0]

# Weighted Mean: Avg Order Value weighted by Discount % (Higher discount orders count more)
# Formula: Sum(Value * Weight) / Sum(Weights)
weighted_mean = np.average(df['Order_Value'], weights=df['Discount_Pct'])

print(f"Mean Order Value: ₹{mean_val:.2f}")
print(f"Median Delivery Days: {median_days} days")
print(f"Mode Rating: {mode_rating} stars")
print(f"Weighted Mean (Value weighted by Discount): ₹{weighted_mean:.2f}")
print("-" * 60)

# ==============================================================================
# MODULE 5: MEASURES OF DISPERSION
# Topics: Range, Variance, Std Dev, IQR, Coefficient of Variation
# ==============================================================================
print("\n--- MODULE 5: DISPERSION ---")

# Range: Max - Min
order_range = df['Order_Value'].max() - df['Order_Value'].min()

# Variance: Average squared deviation
variance = df['Order_Value'].var()

# Standard Deviation: Spread around the mean
std_dev = df['Order_Value'].std()

# Interquartile Range (IQR): 75th percentile - 25th percentile
Q1 = df['Order_Value'].quantile(0.25)
Q3 = df['Order_Value'].quantile(0.75)
iqr = Q3 - Q1

# Coefficient of Variation (CV): Std Dev / Mean (Compare volatility)
cv = (std_dev / mean_val) * 100

print(f"Range: ₹{order_range:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: ₹{std_dev:.2f}")
print(f"IQR: ₹{iqr:.2f}")
print(f"Coefficient of Variation: {cv:.2f}%")
print("-" * 60)

# ==============================================================================
# MODULE 6: SKEWNESS AND KURTOSIS
# Topics: Shape of distribution
# ==============================================================================
print("\n--- MODULE 6: DISTRIBUTION SHAPE ---")

skew = df['Order_Value'].skew()
kurt = df['Order_Value'].kurt()

print(f"Skewness: {skew:.2f} (0 is symmetric, >0 right tail, <0 left tail)")
print(f"Kurtosis: {kurt:.2f} (3 is normal, >3 peaked/heavy tails)")

if abs(skew) < 0.5:
    print("Interpretation: Distribution is approximately symmetric.")
else:
    print("Interpretation: Distribution is skewed.")
print("-" * 60)

# ==============================================================================
# MODULE 7: CORRELATION ANALYSIS
# Topics: Pearson, Spearman
# ==============================================================================
print("\n--- MODULE 7: CORRELATION ---")

# Pearson: Linear relationship
pearson_corr = df['Discount_Pct'].corr(df['Order_Value'], method='pearson')

# Spearman: Rank relationship (Good for ordinal data like Ratings)
spearman_corr = df['Delivery_Days'].corr(df['Customer_Rating'], method='spearman')

print(f"Pearson Correlation (Discount vs Value): {pearson_corr:.2f}")
print(f"Spearman Correlation (Delivery vs Rating): {spearman_corr:.2f}")
print("-" * 60)

# ==============================================================================
# MODULE 8: REGRESSION ANALYSIS
# Topics: Simple Linear Regression, Slope, Intercept
# ==============================================================================
print("\n--- MODULE 8: REGRESSION ANALYSIS ---")

# Preparing data (Reshaping needed for sklearn)
X = df[['Discount_Pct']].values  # Independent Variable
y = df['Order_Value'].values     # Dependent Variable

model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Regression Equation: Order Value = {intercept:.2f} + ({slope:.2f} * Discount%)")
print(f"Interpretation: For every 1% increase in discount, Order Value increases by ₹{slope:.2f}")

# Prediction
pred_val = model.predict([[20]])[0]
print(f"Predicted Order Value at 20% Discount: ₹{pred_val:.2f}")
print("-" * 60)

# ==============================================================================
# MODULE 9 & 10: PROBABILITY & DISTRIBUTIONS
# Topics: Binomial, Poisson, Normal, Independent Events
# ==============================================================================
print("\n--- MODULE 9 & 10: PROBABILITY ---")

# 1. Independent Events P(A and B) = P(A) * P(B)
p_north = len(df[df['Warehouse']=='North']) / n
p_tech = len(df[df['Category']=='Tech']) / n
print(f"P(North Warehouse): {p_north:.2f}")
print(f"P(Tech Category): {p_tech:.2f}")

# 2. Binomial Distribution
# Probability of exactly 2 returns in 10 orders (p = return rate)
p_return = df['Returned'].mean()
prob_binom = stats.binom.pmf(k=2, n=10, p=p_return)
print(f"Binomial: Prob of exactly 2 returns in 10 orders: {prob_binom:.4f}")

# 3. Poisson Distribution
# Avg returns per day = 5. Prob of seeing exactly 8 returns?
prob_poisson = stats.poisson.pmf(k=8, mu=5)
print(f"Poisson: Prob of 8 returns (given avg=5): {prob_poisson:.4f}")

# 4. Normal Distribution
# Prob that an order value is > 3000
prob_norm = 1 - stats.norm.cdf(3000, loc=mean_val, scale=std_dev)
print(f"Normal: Prob of Order Value > 3000: {prob_norm:.4f}")
print("-" * 60)

# ==============================================================================
# MODULE 11: SAMPLING & CLT
# Topics: Random Sampling, Stratified Sampling, Central Limit Theorem
# ==============================================================================
print("\n--- MODULE 11: SAMPLING & CLT ---")

# Simple Random Sample
sample_random = df.sample(n=100)
print(f"Simple Random Sample Mean: {sample_random['Order_Value'].mean():.2f}")

# Stratified Sample (Proportional to Warehouse size)
stratified = df.groupby('Warehouse', group_keys=False).apply(lambda x: x.sample(frac=0.1))
print(f"Stratified Sample Size: {len(stratified)}")

# Central Limit Theorem (CLT) Demonstration
# Taking 500 samples of size 30 and plotting means
sample_means = [df['Order_Value'].sample(30).mean() for _ in range(500)]
plt.figure(figsize=(8,4))
sns.histplot(sample_means, kde=True, color='purple')
plt.title('CLT: Distribution of Sample Means (Should be Normal)')
plt.show()

# Standard Error
std_error = std_dev / np.sqrt(n)
print(f"Standard Error of Mean: {std_error:.2f}")
print("-" * 60)

# ==============================================================================
# MODULE 12: HYPOTHESIS TESTING
# Topics: t-test, p-value, Null Hypothesis
# ==============================================================================
print("\n--- MODULE 12: HYPOTHESIS TESTING ---")
print("H0: No difference in delivery time between North & South warehouses.")
print("H1: North warehouse delivery time is different.")

north_days = df[df['Warehouse']=='North']['Delivery_Days']
south_days = df[df['Warehouse']=='South']['Delivery_Days']

# Independent t-test
t_stat, p_val = stats.ttest_ind(north_days, south_days)

print(f"T-statistic: {t_stat:.2f}")
print(f"P-value: {p_val:.5f}")

if p_val < 0.05:
    print("Result: Reject H0. There is a significant difference.")
else:
    print("Result: Fail to reject H0. No significant difference found.")
print("-" * 60)

# ==============================================================================
# MODULE 13: PROJECT CONCLUSION
# ==============================================================================
print("\n--- MODULE 13: MINI PROJECT CONCLUSION ---")
print("1. Data generated and cleaned.")
print("2. Visualizations identified 'North' warehouse as an outlier in delivery time.")
print("3. Regression showed positive correlation between Discount and Order Value.")
print("4. Hypothesis testing confirmed 'North' warehouse is significantly slower.")
print("Analysis Complete.")
