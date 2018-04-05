# UCSB Intelligent Machine Vision Course 

## Development Environment Setup

We support Anaconda (https://www.anaconda.com/download) using Python 3.6. If you want to isolate installed packages from the rest of your Python system, make sure to install Anaconda for the local user only and do not add conda to the path (this is a check-box option during installation). A conda environment will be used to further isolate things.

### Create Conda Environment
The following creates a conda environment called `imv`:<br>
`conda create --name imv`<br>


### Activate Conda Environment
You will need to activate your environment each time you open a new terminal.<br>
<br>
MacOS and Ubuntu:<br>
`source activate imv`<br>
<br>
Windows 10:<br>
`activate imv`<br>
<br>
This isolates all `pip` or `conda` installs from your other environments or from your system-level Python installation.

### Install packages
Ensure the `imv` environment is activated, then:<br>
`conda install -c conda-forge opencv`<br>
