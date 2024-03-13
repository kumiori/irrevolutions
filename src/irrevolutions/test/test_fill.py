import matplotlib.pyplot as plt
import numpy as np

# Generate some example data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the line plot with round markers
plt.plot(x, y1, marker='o', label='Curve 1')
plt.plot(x, y2, marker='o', label='Curve 2')

# Shade the area between the two curves
plt.fill_between(x, y1, y2, where=(y1 >= y2), interpolate=True, color='blue', alpha=0.3, label='Shaded Area')

# Add labels and legend
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot with Shaded Area')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()
