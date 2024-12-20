# auto-path





## How to run the project

### Prerequisites (run before any other steps)

All examples apply for Linux systems and Python3.12.

1. Install Python on your PC

    E.g. Linux already contains it, for Windows you can download the installer from the official website.

2. Install virtualenv and tkinter package in Python

    E.g. for Ubuntu run
    
    `sudo apt install python3-venv`
    `sudo apt install python3-tk`

3. Setup the virtual environment

    E.g. for Ubuntu, run `python3 -m venv .venv` to create a virtual environment inside .venv folder.
    Then, `source .venv/bin/activate` to activate the environment.
    Then, install dependencies: `pip install -r requirements.txt `

### A. Generate a terrain

Code is in [examples_a.py](examples_a.py)

Simply run in your terminal:

`python3 examples_a.py`

### B. Sample on generating the figures in the article

Code is in [generate_figures.py](generate_figures.py)

Simply run in your terminal:

`python3 generate_figures.py`


### C. Sample of outputting to CSV files

Code is in [generate_csv.py](generate_csv.py)

Simply run in your terminal:

`python3 generate_csv.py`


## Authors provide two examples (terrain and generated path)

### 1 First Example

_**Info:** All CSV files have 3 columns: coordinate 1, coordinate 2, height_

We start generating the terrain using this config:

```python
# define terrain configuration
config = TerrainGeneratorConfig(
    seed=0, GRID_SIZE=(100, 100),
    scaling_argument=(4, 4),
    height_interval=(100, 120),
    height_delta=3
)
# pass configuration to a new experiment
e = Experiment(config=config)
```
We continue by setting the *start* and *target* points of the road as:

```python
# setup the start and end of the road to generate
start, target = (
    (5, 5),
    (config.GRID_SIZE[0] - 5, config.GRID_SIZE[1] - 5)
)
(
    e.area_sections.orig_area.start,
    e.area_sections.orig_area.target
) = (start, target)

```

#### To generate the results on your own you can run 

```python
python generate_csv.py
# ^ you can update generate_csv.py yourself, and change the "TerrainGeneratorConfig", "start" or "target" values
```

Terrain: [terrain.csv](./terrain.csv)

![Screenshot from 2025-05-08 09-52-23](https://github.com/user-attachments/assets/24188ca8-4b34-4c5e-be4c-e1930ed5c8b4)


Path: [path.csv](./path.csv)

![Screenshot from 2025-05-08 09-52-13](https://github.com/user-attachments/assets/216688ea-dfd4-4173-a595-a32fd7cbb0e3)


### 2 Second Example

_**Info:** All CSV files contain three columns: coordinate 1, coordinate 2, height._

We start generating the terrain using this config:

```python
# define terrain configuration
config = TerrainGeneratorConfig(
    seed=7, GRID_SIZE=(60, 60),
    scaling_argument=(2, 2),
    height_interval=(320, 336),
    height_delta=2
)
# pass configuration to a new experiment
e = Experiment(config=config)
```

We continue by setting the *start* and *target* points of the road as:

```python
# setup the start and end of the road to generate
start, target = (
    (15, 15),
    (config.GRID_SIZE[0] - 15, config.GRID_SIZE[1] - 15)
)
(
    e.area_sections.orig_area.start,
    e.area_sections.orig_area.target
) = (start, target)
```

#### To generate the results on your own you can run 

```python
python generate_csv.py
# ^ you can update generate_csv.py yourself, and change the "TerrainGeneratorConfig", "start" or "target" values
```

Terrain: [terrain2.csv](./terrain2.csv)

![Screenshot from 2025-05-08 10-18-06](https://github.com/user-attachments/assets/111bcece-4f17-49b7-b2cd-12fb8f012b57)


Path: [path2.csv](./path2.csv)

![Screenshot from 2025-05-08 10-17-13](https://github.com/user-attachments/assets/b04b9bd3-9144-492c-a810-fa7610708551)


