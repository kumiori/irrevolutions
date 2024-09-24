import subprocess

import numpy as np

# List of parameters

# parameters_list = np.logspace(-8, -1, 8)
parameters_list = np.arange(2, 10, 1)

# Iterate through the parameters and launch the script with each parameter
for param in parameters_list:
    # Command to run the Python script with a parameter
    command = ["python3", "traction-parametric.py", "-n", str(param)]

    try:
        # Execute the command
        print(command)
        subprocess.run(command, check=True)
        print(f"Script executed successfully with parameter: {param}")
    except subprocess.CalledProcessError as e:
        # Handle any errors or exceptions
        print(f"Error executing script with parameter {param}: {e}")
