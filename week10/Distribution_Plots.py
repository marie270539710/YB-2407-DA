import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

tips = sns.load_dataset('tips')

# Load your CSV file into a pandas DataFrame
df = pd.read_csv("week10/Sample_Data_for_Activity.csv")  # Replace with your CSV path

# Use Seaborn for visualization
#sns.histplot(df["Normal_Distribution"])  
#sns.displot(df['Normal_Distribution'], kde=True)
sns.displot(df['Normal_Distribution'], kde=True, bins=200)

#sns.jointplot(x='Normal_Distribution', y='Exponential_Distribution', data=df, kind='scatter')
sns.jointplot(x='Normal_Distribution', y='Exponential_Distribution', data=df, kind='reg')
#sns.jointplot(x='Normal_Distribution', y='Exponential_Distribution', data=df, kind='hex')

#sns.jointplot(x='Normal_Distribution', y='Exponential_Distribution', data=df, hue='Poisson_Distribution')
#sns.jointplot(x='Normal_Distribution', y='Exponential_Distribution', data=df, hue='Poisson_Distribution', kind='kde')

sns.pairplot(df)

# Show the plot
plt.show()

#tips.head()

#tips.info()