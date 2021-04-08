# TOPH 

TOPH (TOols for PSF Homogenization, in keeping with the tradition of terrible astronomy acronyms) performs spatially-variable point spread function matching. The main routines are:

* `run_toph.py` -- calls the main program: convolve_image.py.
* `toph_params.py` -- file for user-supplied parameters.

# Installation

To get this repo running:

* Install Python 3.  You can find instructions [here](https://wiki.python.org/moin/BeginnersGuide/Download).
* Create a [virtual environment](https://docs.python.org/3/library/venv.html).
* Clone this repo with `git clone git@github.com:dataquestio/analytics_pipeline.git`
* Get into the folder with `cd TOPH`
* Install the requirements with `pip install -r toph_dependencies.txt`

# Usage

* Update `toph_params.py` with the appropriate paramater information.
* In `TOPH/toph_scripts/` type `python run_toph.py`.

You should see output on the command line. Scripts for diagnostic plots coming soon!