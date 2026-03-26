import subprocess

import numpy as np

# List of parameters

# parameters_list = np.logspace(-8, -1, 8)
# parameters_list = [2e-3, 5e-3, 1e-2, 2e-2, 5e-2]
# parameters_list = [2e-4, 5e-4, 3e-3, 2e-3, 5e-3]
parameters_list = np.logspace(-8, -1, 8)
parameters_list = [2e-1, 3e-1, 4e-1, 5e-1]
parameters_list = [6e-1, 1, 1.3]
parameters_list = np.logspace(-5, -1, 8)

# Iterate through the parameters and launch the script with each parameter
for param in parameters_list:
    # Command to run the Python script with a parameter
    command = ["python3", "traction-parametric.py", "-s", str(param)]
    command = ["python3", "traction-parametric.py", "-s", str(param), "--model", "at1"]

    try:
        # Execute the command
        print(command)
        subprocess.run(command, check=True)
        print(f"Script executed successfully with parameter: {param}")
    except subprocess.CalledProcessError as e:
        # Handle any errors or exceptions
        print(f"Error executing script with parameter {param}: {e}")
