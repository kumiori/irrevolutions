import os
import numpy as np

# Example parameters
L = 10  # Replace with actual length
num_points = 10  # Replace with actual number of points
num_modes = 1  # Replace with actual number of modes
num_time_steps = 10  # Replace with actual number of time steps


# Example: Assuming `mode_shapes_data` is your existing dictionary
mode_shapes_data = {
    "time_steps": [],
    "point_values": {
        "x_values": np.linspace(0, L, num_points),
    },
}

# Create a list to store flattened data
flattened_data = []

# Example time-stepping loop
for current_time_step in range(num_time_steps):
    # Save the current time step
    mode_shapes_data["time_steps"].append(current_time_step)
    print(f"Processing time step {current_time_step}...")
    # Initialize mode-specific fields
    for mode in range(1, num_modes + 1):
        field_1_values_mode = np.random.rand(num_points)  # Replace with actual values
        field_2_values_mode = np.random.rand(num_points)  # Replace with actual values

        # Append mode-specific fields to the data structure
        mode_key = f"mode_{mode}"
        mode_shapes_data["point_values"][mode_key] = {
            "field_1": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("field_1", []),
            "field_2": mode_shapes_data["point_values"]
            .get(mode_key, {})
            .get("field_2", []),
        }
        mode_shapes_data["point_values"][mode_key]["field_1"].append(
            field_1_values_mode
        )
        mode_shapes_data["point_values"][mode_key]["field_2"].append(
            field_2_values_mode
        )

# Save the mode_shapes_data dictionary to a file (e.g., using np.savez or
# pickle)
np.savez(
    os.path.join(os.path.dirname(__file__), "data/mode_shapes_data.npz"),
    **mode_shapes_data,
)
