import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("../data/cleaned_insurance_data.csv")


numeric_cols = df.select_dtypes(include=['int64','float64']).columns

for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(col, "->", outliers.shape[0], "outliers")


#Store Original Data 
original_data = df['policy_annual_premium'].copy()



plt.figure(figsize=(6,4))
sns.boxplot(x=df['policy_annual_premium'])
plt.title("Boxplot of Policy Annual Premium")
plt.show()


plt.figure(figsize=(6,4))
sns.histplot(df['policy_annual_premium'], kde=True)
plt.title("Distribution of Policy Annual Premium")
plt.show()

Q1 = df['policy_annual_premium'].quantile(0.25)
Q3 = df['policy_annual_premium'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)



outliers = df[(df['policy_annual_premium'] < lower_bound) | 
              (df['policy_annual_premium'] > upper_bound)]

print("Number of Outliers:", outliers.shape[0])

# Log Transformation
df['policy_annual_premium'] = np.log1p(df['policy_annual_premium'])

plt.figure(figsize=(6,4))
sns.boxplot(x=df['policy_annual_premium'])
plt.title("After Log Transformation")
plt.show()

#Before and After Comparison
fig, ax = plt.subplots(1,2, figsize=(12,4))

sns.histplot(original_data, ax=ax[0], kde=True)
ax[0].set_title("Before Transformation")

sns.histplot(df['policy_annual_premium'], ax=ax[1], kde=True)
ax[1].set_title("After Log Transformation")

plt.show()  


# Save the updated dataset after outlier transformation
df.to_csv("processed_insurance_data.csv", index=False)

print("Dataset saved successfully.")
