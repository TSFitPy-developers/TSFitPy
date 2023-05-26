# Turbospectrum Spectral Fitting with Python (TSFitPy)
<!--# AKA PODRACES (Pipeline for the Objective Determination of Realistically Accurate Characteristics and Elements of Stars)-->

TSFitPy is a pipeline designed to determine stellar abundances and atmospheric parameters through the use of Nelder-Mead (simplex algorithm) minimization. It calculates model spectra "on the fly" while fitting instead of using a more sophisticated method that relies on training neural networks such as the method used by the full SAPP used for much larger datasets. Using this method allows the pipeline to gain flexibility in the stellar physics used at the expense of computation time, which makes it useful for small datasets of ~100 stars and fewer.

To use TSFitPy, you will need a working Turbospectrum (TS) installation of the latest version, which has the capability to compute NLTE line profiles as well as calculate specified spectral windows instead of a full spectrum for a given range. TSFitPy has not been tested on older versions of TS. The latest version of TS can be found here: https://github.com/bertrandplez/Turbospectrum_NLTE

The code requires at least version Python 3.7. It also makes use of fortran programs, which will need to be compiled on the 
user's machine. The Python packages needed are as follows (they should all be installable via "pip install"):
- Numpy
- Scipy (at least 1.7.1)
- Dask (installed using pip install dask[complete])
- Pandas (only for plotting)
- Matplotlib (only for plotting)

## Installation

- Clone or download code to your favourite directory
- Download actual [TurboSpectrum fortran code](https://github.com/bertrandplez/Turbospectrum_NLTE) and put it into folder `TSFitPy/turbospectrum/`
- Compile TS fortran code using the make file in `turbospectrum/exec/` (or in `turbospectrum/exec-gf/` if using the gnu compiler)
- Copy fortran files (can copy everything in unsure) from `TSFitPy/turbospectrum/interpolator/` to `TSFitPy/scripts/model_interpolators/`
- Run `TSFitPy/scripts/compile_fortran_codes.py` to compile model interpolators
- Download all desired linelists and put them into `TSFitPy/input_files/linelists/linelist_for_fitting/`
  - Example VALD lines are included in the `TSFitPy/input_files/linelists/linelist_vald/`, which you can move to the `TSFitPy/input_files/linelists/linelist_for_fitting/`
  - **ALTERNATIVELY** but **required** for the NLTE Gaia-ESO linelists are provided [here](https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/) in the file `nlte_ges_linelist` (wavelength ranges: 4200-9200 Å)
  - Additional linelists to include are vald ones (3700-3800, 3800-4200, 9200-9300, 9300-9800) that extend the wavelength regime of the Gaia-ESO linelist
  - Molecular linelists may also be important. They are found in the same link as the Gaia-ESO linelist in the folder `molecules-420-920nm`
  - **IMPORTANT**: ALL files in the `TSFitPy/input_files/linelists/linelist_for_fitting/` are used, so do not use BOTH Gaia-ESO and VALD data from same wavelength ranges
- Download desired atmospheric models and put them into the `TSFitPy/input_files/model_atmospheres/` in either `1D` or `3D` folder
  - 1D MARCS standard composition models are included [here](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) in the folder `atmospheres`
  - 3D averaged STAGGER models can be included in the `3D` folder as well
- If desired to use NLTE data, then one needs to provide departure coefficient files. They can be downloaded from [here](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) in the `dep-grids` folder.
  - The size of each file is big (anywhere from a few GB up to a few dozen GB), so only download relevant files
  - In the relevant element file you will find several files, for 1D MARCS you will need:
    - atom.ELEMENT_AND_NUMBER
    - auxData_ELEMENT_MARCS_DATE.txt
    - NLTEgrid_ELEMENT_MARCS_DATE.bin
  - The same idea applies for 3D averaged STAGGER atmospheres:
    - atom.ELEMENT_AND_NUMBER
    - auxData_ELEMENT_STAGGERmean3D_DATE_marcs_names.txt
    - NLTEgrid_Ba_STAGGERmean3D_DATE.bin
  - The naming might vary, but use MARCS for 1D models, use STAGGERmean3D (with marcs_names!) for mean STAGGER models
  - For each element, download, unzip and place auxData and NLTEgrid into their own folder (e.g. `TSFitPy/input_files/nlte_data/Ba/`)
  - Put all relevant atom files into `TSFitPy/input_files/nlte_data/model_atoms/`
  - Important note: even if you want to only fit one specific element in NLTE, you will always need Fe NLTE data, so download that one as well
- Some default linemask files are already provided in the script; linemask are wavelength ranges where line will be fitted (i.e. left and right sides of the line)
  - They are located in the `TSFitPy/input_files/linemask_files/` and separated into individual folders
  - Each folder contains two files: `ELEMENT-lmask.txt` and `ELEMENT-seg.txt`
  - You can also create your own linemasks.
  - To create your own linemask:
    - Create new textfile (naming doesn't matter)
      - First column: wavelength center of the line (no need to be exact, it is only used in printing)
      - Second column: left side of the line where it is fitted (i.e. include wings as well)
      - Third column: right side of the line where it is fitted
    - You can add comments using `;`

## Usage

- There are three main steps to take for every fit: get normalised spectra, create corresponding fitlist and change the configuration (config) file
- As an example, we are going to use provided sample spectrum and change config file to fit the Sun
- Take spectrum from `TSFitPy/input_files/sample_spectrum/` and put it into desired folder, such as ``TSFitPy/input_files/observed spectra/`
- It is recommended to create a separate config file for each run/set of stars. This allows to quickly refit the same sample of stars without recreating config file each time
- Copy and paste the existing `TSFitPy/input_files/tsfitpy_input_configuration.txt` and call it something like `TSFitPy/input_files/tsfitpy_input_configuration_sun_test.txt`
- The config file should already be ready for a test run, but here is the reference breakdown if needed
  - `turbospectrum_compiler` specifies the compiler (intel or gnu). Location of turbospectrum is expected at `TSFitPy/turbospectrum/`
  - Next few lines specify the paths. Default paths are relative to the `TSFitPy/scripts/TSFitPy.py`, but it is possible to change paths if you want to keep your data in a separate folder (e.g. it can be useful if sharing data on a cluster)
  - `debug` can be used for debugging code. 0 is best for normal fits, 1 outputs some extra information during the Python fitting, 2 outputs full TS fortran information (a lot of info and much slower fit)
  - `atmosphere_type` 1D or 3D: MARCS is 1D, STAGGER mean are 3D models
  - `mode` specifies fitting mode
    - `all` fits all lines within the linemask at the same time. Advantage: faster. Disadvantage: cannot know whether any specific line has good or bad fit. Not recommended
    - `lbl` fits all lines within the linemask one line at a time. Advantage: get full info for each line with separate abundance, macroturbulence etc. Can also fit microturbulence (not very well though?) Disadvatage: slower.
    - `teff` fits specified line by changing temperature, not abundance. Recommended use: use element H and include NLTE for H and Fe.
  - `include_molecules` is whether you want molecules in your spectra. Fitting can be faster without them (useful when testing?). Recommended: yes, unless molecules are not expected in the spectra.
  - `nlte` whether want to have NLTE or not. Elements to include with NLTE are written below
  - `fit_microturb`, `fit_macroturb` Yes/No/Input depending on if you want to fit them or not. If "no", then microturbulence is calculated based on empirical relation (based on teff, logg, [Fe/H]) and works rather well for FGK-type stars. If Input, it is possible to input microturbulence in the fitlist later. If macroturbulence is "no", then constant one will be applied to all stars (chosen below). If Input, then each star can be given one in the fitlist later on
  - `fit_rotation` Yes/No as well. Yes - fits rotation (not recommended to use togehter with macroturbulence). No - takes constant one for all below.
  - `element` which element to fit. Normally one would fit one element at a time, but it is possible to fit several elements at once using the same linemask (e.g. blended line). If you want to fit abundance for different lines, then you need to fit one element at a time
  - `linemask_file` is the path in the `TSFitPy/input_files/linemask_files/` from where the linemask is taken
  - `departure_coefficient_binary`, `departure_coefficient_aux` are paths in the `TSFitPy/input_files/nlte_data/` for the files that need NLTE data (only if NLTE is fitted). You can put several of them by separating them with a space. Order should be the same as the `element` above. If Fe is not fitted, it has to be included regardless (it is also put last).
  - `model_atom_file` are names of model atoms in the `TSFitPy/input_files/nlte_data/model_atoms/` with the same order as two lines above
  - `wavelength_minimum` and `wavelength_maximum` are **ONLY** used if using `mode = all`, which specifies the ranges of the fitted spectra. Otherwise ignored
  - `wavelength_delta` is the synthetic generated `wavelength_delta`. Try not to have it less than observed spectra, but too small will result in slow fitting. Recommended as a start: `0.005`
  - `resolution` is resolution of teh spectra. 0 is no convolution based on the resolution
  - `macroturbulence` is default macroturbulence for all stars if `fit_macroturb = No`
  - `rotation` is default macroturbulence for all stars if `fit_rotation = No`
  - `input_file` is the name of the fitlist file that will be explained later
  - `output_file` is the name of the output file, usually expected to be `output` for the plotting tool
  - `workers` are the amount of workers to use for the Dask multiprocessing package. Do not set higher than the amount of CPU cores that you have. 1 works best for debugging.
  - init/input
  - bounds
  - guess
  - experimental
- An example of `fitlist` file is added


Once these folders are set, you can begin fitting with TSFitPy. Set up the parameters of your fit using the "tsfitpy_input_configuration.txt" file, place normalized observed spectra for fitting in the folder "observed_spectra" in input_files, and update the "fitlist" file to list the spectra to be fit and their parameters. The folder "examples" provides some examples of various fitting methods as well as sample input files, output files, and terminal outputs. These examples were done using the Gaia-ESO linelists provided at the link above.


Recently added changes:
- Full code refactoring. Faster fitting
- Multiprocessing support using Dask. Change number of workers to number of requested CPUs (setting to 1 does not turn on multithreading) in the TSFitPy.py (at the bottom). Change login_node_access to your node login. Then check the status of the Dask by checking the output of the slurm script (should be something like ssh -N -L PORT_NUMBER:node03:PORT_NUMBER supercomputer.university.edu), run that command on your local terminal. Now you should be able to connect to Dask dashboard via "localhost:8787/status" (more about it here: https://github.com/dask/dask/discussions/7480).
- Fit lbl with macro and rotation
- More configuration so that the TSFitPy script does not need to be changed for anything basically
- Ability to fit much quicker at a cost of exact fitting (at least x4-5 times faster). Outputs a long series of chi-squares depending on the input guess for metallicity/abundance. The curve can be plotted to find lowest point -> lowest chi-squared might correspond to the correct metallicity? Use "lbl_quick" for this method.


