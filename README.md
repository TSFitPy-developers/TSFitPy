# Development branch of TSFitPy

Changes:
- Full code refactoring. Faster fitting
- Multiprocessing support using Dask. Change number of workers to number of requested CPUs (setting to 1 does not turn on multithreading) in the TSFitPy.py (at the bottom). Change login_node_access to your node login. Then check the status of the Dask by checking the output of the slurm script (should be something like ssh -N -L PORT_NUMBER:node03:PORT_NUMBER supercomputer.university.edu), run that command on your local terminal. Now you should be able to connect to Dask dashboard via "localhost:8787/status" (more about it here: https://github.com/dask/dask/discussions/7480).
- Fit lbl with macro and rotation
- More configuration so that the TSFitPy script does not need to be changed for anything basically
- Ability to fit much quicker at a cost of exact fitting (at least x4-5 times faster). Outputs a long series of chi-squares depending on the input guess for metallicity/abundance. The curve can be plotted to find lowest point -> lowest chi-squared might correspond to the correct metallicity? Use "lbl_quick" for this method.

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
  - **ALTERNATIVELY** Gaia-ESO linelists are provided [here](https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/) in the file `nlte_ges_linelist` (wavelength ranges: 4200-9200 Å)
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
    - Each line needs to be included within a corresponding segment
      - Segment are wavelengths where the spectra is computed for the line, such that the wings from other lines/blends are also included in the actual line
    - Rule of thumb: line center +/- 3-5 Å is one segment
    - Segment can contain several or no lines:
      - First column: wavelength to the left of the line
      - Second column: wavelength to the right of the line.
      - Feel free to merge segments together

## Usage

- There are two main steps to take for every fit: get normalised spectra and change the configuration (config) file
- As an example, we are going to use provided sample spectrum and change config file to fit the Sun
- Take spectrum from `TSFitPy/input_files/sample_spectrum/` and put it into desired folder, such as ``TSFitPy/input_files/observed spectra/`
- It is recommended to create a separate config file for each run/set of stars. This allows to quickly refit the same sample of stars without recreating config file each time
- Copy and paste the existing `TSFitPy/input_files/tsfitpy_input_configuration.txt` and call it something like `TSFitPy/input_files/tsfitpy_input_configuration_sun_test.txt`
- The config file should already be ready for a test run, but it is worth going through it as well
  - `turbospectrum_compiler` specifies the compiler (intel or gnu). Location of turbospectrum is expected at `TSFitPy/turbospectrum/`
  - Next few lines specify the paths. Default paths are relative to the `TSFitPy/scripts/TSFitPy.py`, but it is possible to change paths if you want to keep your data in a separate folder (e.g. it can be useful if sharing data on a cluster)
  - debug

All of the fortran codes are compilable either with a gnu or ifort compiler. In the scripts folder, there is a Python script titled "compile_fortran_codes.py". Running this code should compile all of the necessary codes needed for the main TSFitPy pipeline. It makes use of the OS Python package.

TSFitPy uses a system of relative directories to run Turbospectrum and know where to find various inputs that control the fitting procedure. These directories need to be preserved in order to run correctly. The main directories in use are input_files, output_files, scripts, turbospectrum, and the various folders within. A working Turbospectrum installation needs to be installed in the "turbospectrum" directory so that the executive files for Turbospectrum are in a directory "turbospectrum/exec/" (or in "turbospectrum/exec-gf/" if using the gnu compiler).

Other directories that require user input are the "model_atmospheres" and "linelists" directories in input_files. The model_atmospheres has two directories, one called "1D" for MARCS model atmospheres (include reference), and one called "3D" for average 3D model atmospheres (include info about STAGGER grid). A zip file of the available standard composition MARCS models is provided at https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/ for your convenience. A list of all model atmospheres in the folder called "model_atmosphere_list.txt" also needs to be in each folder. An example is provided to show the formatting.

The linelist/s used by Turbospectrum should be put in the folder "linelists/linelist_for_fitting/". Some example linelists are included that were made from the VALD3. In addition, various atomic and molecular linelists used by the Gaia-ESO survey can be found here https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/. The atomic lists have been cross-matched with the model atoms used for NLTE to identify transition levels for the lines and can therefore be used for NLTE computations with Turbospectrum. We have supplemented these lists with lines pulled from the VALD3 for lines falling outside of the Gaia-ESO range (4200 - 9200 AA). These files cannot end in ".txt".

Finally, if you wish to fit using NLTE, you'll need the relevant NLTE data. This includes binary files of departure coefficients, auxiliary text files that tell the model interpolators how to read these files, and model atom files. These files are too large to keep on Github (especially the binary files), so those available to the public can be found at https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/. The information then needs to be stored in the relevant directories under "input_files/nlte_data". This directory also has model atmospheres (both average 3D from the STAGGER grid and the 1D MARCS standard composition models).

Once these folders are set, you can begin fitting with TSFitPy. Set up the parameters of your fit using the "tsfitpy_input_configuration.txt" file, place normalized observed spectra for fitting in the folder "observed_spectra" in input_files, and update the "fitlist" file to list the spectra to be fit and their parameters. The folder "examples" provides some examples of various fitting methods as well as sample input files, output files, and terminal outputs. These examples were done using the Gaia-ESO linelists provided at the link above.

