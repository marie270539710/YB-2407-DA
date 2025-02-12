import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
levels = ['Gold', 'Silver', 'Bronze']
customer_levels = np.random.choice(levels, 200, p=[0.3, 0.4, 0.3])
level_counts = {level: np.sum(customer_levels == level) for level in levels}
plt.figure(figsize=(6,6))
plt.bar(level_counts.keys(), level_counts.values(), color=['gold', 'silver', 'brown'])
plt.xlabel("Customer Level")
plt.ylabel("Number of Customers")
plt.title("Customer Segmentation in Retail Industry")

# Save
plt.savefig("week9/customer_segmentation_in_retail_industry.png")
plt.show()

# Q: What is the trend in the following code?
# A: 
# Gold and Bronze level customers are equally distributed, with Silver level customers making up the largest group.
# According to this trend, the most frequent clients in this retail sector are Silver level customers.