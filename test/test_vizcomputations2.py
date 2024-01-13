
import matplotlib.pyplot as plt
import numpy as np
# Example list of computations, where each computation is a list of partial time quotas
computations = [
    [10, 20, 15, 5, 30],
    [15, 10, 25, 8, 20],
    [5, 15, 10, 25, 30]
]

# Create a list of tasks (assuming the same tasks for all computations)
tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5']

# Convert the list of computations into a NumPy array for easier manipulation
computations = np.array(computations)

# Calculate the number of computations and the width of each group of bars
num_computations = len(computations)
bar_width = 0.4  # Width of each bar
index = np.arange(len(tasks))  # x-axis values for the tasks

# Create a figure and axis
fig, ax = plt.subplots()

# Initialize the bottom for each bar (for stacking)
bottom = np.zeros(len(tasks))

# Create stacked bars for each computation
for i in range(num_computations):
    ax.bar(index, computations[i], bar_width, label=f'Computation {i+1}', bottom=bottom)
    bottom += computations[i]

# Set labels and title
ax.set_xlabel('Tasks')
ax.set_ylabel('Time Quotas (minutes)')
ax.set_title('Stacked Time Quotas for Computations')

# Set x-axis ticks and labels
ax.set_xticks(index)
ax.set_xticklabels(tasks)

# Show the legend
plt.legend()

# Show the chart
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()