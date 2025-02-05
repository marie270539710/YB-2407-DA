import matplotlib.pyplot as plt

# Data
study_hours = [1,2,3,4,5,6,7,8,9,10]
test_scores = [50,55,65,70,72,78,85,88,92,95]

# Create scatter plot
plt.figure(figsize=(5, 5))
plt.scatter(study_hours, test_scores, color='purple', marker='*')
plt.xlabel("Study Hours")
plt.ylabel("Test Scores")
plt.title("Study Hours vs Test Scores")
plt.grid(True)

# Save
plt.savefig("week9/study_hours_vs_test_scores.png")
plt.show()