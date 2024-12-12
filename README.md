# Optimal ambition in business, politics, and life
This repository houses the code for the manuscript "Optimal ambition in business, politics, and life"

## Installation instructions
Use `git clone` to download this repository either via HTTPS

```shell
git clone https://github.com/kathlandgren/ambition.git
```

or SSH

```shell
git clone git@github.com:kathlandgren/ambition.git
```

### Create Python virtual environment

Create a Python virtual environment in the root of the publishorcomparish directory:

  ```shell
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Contents

The repository is organized into the following directories:
- 00_analytical_function_setup/ contains the functions used in the analytical expression for expected average reward
- 01_simulation_code/ contains the modules and scripts used in running the simulations
- folders 02 through 05 contain the code used to generate main manuscript figures
- folders 06 through 08 contain the code used to generate supplementary information figures.
