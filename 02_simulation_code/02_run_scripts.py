# This script runs all simulation scripts in specified file location.
import subprocess

def run_script(script_path):
    try:
        subprocess.run(["python3", script_path], check=True)
        print(f"Script '{script_path}' executed successfully.")
    except subprocess.CalledProcessError:
        print(f"Error executing script '{script_path}'.")

# base string convention for naming simulation files
base_string = "01_sample_simulation_{}.py"
# specify the range of simulation files to run
numbers = list(range(0, 1))

scripts_to_run = [base_string.format(num) for num in numbers]

# Run each script in the list
for script_path in scripts_to_run:
    run_script(script_path)