# TOPH [![BCH compliance](https://bettercodehub.com/edge/badge/jacqdanso/TOPH?branch=main&token=c71d7fd00b87246bc0dd9ce8469379d245b7096d)](https://bettercodehub.com/) [![Build Status](https://travis-ci.com/jacqdanso/TOPH.svg?token=qzXtwAJm34Dyam53QEre&branch=main)](https://travis-ci.com/jacqdanso/TOPH)

TOPH (TOols for PSF Homogenization, in keeping with the tradition of terrible astronomy acronyms) performs spatially-variable point spread function matching. The main routines are:

* `run_toph.py` -- calls the main program: convolve_image.py.
* `toph_params.py` -- file for user-supplied parameters.

# Installation

To get this repo running:

* Install Python 3.  You can find instructions [here](https://wiki.python.org/moin/BeginnersGuide/Download).
* Create a [virtual environment](https://docs.python.org/3/library/venv.html).
* Clone this repo with `git clone https://github.com/jacqdanso/TOPH.git`
* Get into the folder with `cd TOPH`
* Install the requirements with `pip install -r requirements.txt`
* You will also need an [IDL](https://www.l3harrisgeospatial.com/Software-Technology/IDL) installation

# Usage

* Update `toph_params.py` with the appropriate paramater information.
* In `TOPH/toph_scripts/` type `python run_toph.py`.

You should see output on the command line. Documentation coming soon!
