# Cool Guys Commit to Main

## Setup
Install [miniconda](https://docs.anaconda.com/miniconda/) on your machine. Inside the miniconda terminal, navigate to the `orbit-cola/` repo and run the following commands

Create the environment named orbit_cola

```conda env create -f environment.yml```

Activate the environment

```conda activate <env_name> python src/main.py```

Remove the environment

```conda remove -n <env_name> --all```

Update environment

```conda env update --name <env_name> --file environment.yaml```


## Data
[TLE Archive](https://celestrak.org/NORAD/archives/)