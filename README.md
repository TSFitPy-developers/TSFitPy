# Turbospectrum Spectral Fitting with Python (TSFitPy)
<!--# AKA PODRACES (Pipeline for the Objective Determination of Realistically Accurate Characteristics and Elements of Stars)-->

TSFitPy is a pipeline designed to determine stellar abundances and atmospheric parameters through the use of Nelder-Mead (simplex algorithm) minimization. It calculates model spectra "on the fly" while fitting instead of using a more sophisticated method that relies on training neural networks such as the method used by the full SAPP used for much larger datasets. Using this method allows the pipeline to gain flexibility in the stellar physics used at the expense of computation time, which makes it useful for small datasets of ~100 stars and fewer.

To use TSFitPy, you will need a working Turbospectrum (TS) installation of the latest version, which has the capability to compute NLTE line profiles as well as calculate specified spectral windows instead of a full spectrum for a given range. TSFitPy has not been tested on older versions of TS. The latest version of TS can be found here: https://github.com/bertrandplez/Turbospectrum_NLTE

The code requires at least version Python 3.7. It also makes use of fortran programs, which will need to be compiled on the 
user's machine. The Python packages needed are as follows (they should all be installable via "pip install"):
- Numpy
- Scipy (at least 1.7.1)
- Dask (installed using pip install dask[complete])
- Pandas
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
  - **IMPORTANT**: ALL files in the `TSFitPy/input_files/linelists/linelist_for_fitting/` are used, so do NOT use BOTH Gaia-ESO and VALD data from same wavelength ranges
- Download desired atmospheric models and put them into the `TSFitPy/input_files/model_atmospheres/` in either `1D` or `3D` folder and unzip them
  - 1D MARCS standard composition models are included [here](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) in the folder `atmospheres/marcs_standard_comp.zip`
  - 3D averaged STAGGER models can be included in the `3D` folder as well (same link, `atmospheres/average_stagger_grid_forTSv20.zip`)
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

## Usage for fitting

- There are three main steps to take for every fit: get normalised spectra, create corresponding fitlist and change the configuration (config) file
- As an example, we are going to use provided sample spectrum and change config file to fit the Sun
- Take spectrum from `TSFitPy/input_files/sample_spectrum/` and put it into desired folder, such as `TSFitPy/input_files/observed spectra/`
  - One can also supply error as third column, but it is not required (otherwise it will be set to 1)
  - Error is sigma, i.e. standard deviation, not variance
  - Error is used in chi2 calculation
- It is recommended to create a separate config file for each run/set of stars. This allows to quickly refit the same sample of stars without recreating config file each time
- Copy and paste the existing `TSFitPy/input_files/tsfitpy_input_configuration.cfg` and call it something like `TSFitPy/input_files/tsfitpy_input_configuration_sun_test.cfg`
- The config file should already be ready for a test run, but here is the reference breakdown if needed
  - [turbospectrum_compiler]:
    - `compiler` specifies the compiler (intel or gnu). Location of turbospectrum is expected at `TSFitPy/turbospectrum/`
  - [MainPaths]
    - Next few lines specify the paths. Default paths are relative to the `TSFitPy/scripts/TSFitPy.py`, but it is possible to change paths if you want to keep your data in a separate folder (e.g. it can be useful if sharing data on a cluster)
  - [FittingParameters]
    - `atmosphere_type` 1D or 3D: MARCS is 1D, STAGGER average are 3D models
    - `mode` specifies fitting mode
      - `all` fits all lines within the linemask at the same time. Advantage: faster. Disadvantage: cannot know whether any specific line has good or bad fit. Not recommended
      - `lbl` fits all lines within the linemask one line at a time. Advantage: get full info for each line with separate abundance, macroturbulence etc. Can also fit microturbulence (not very well though?) Disadvatage: slower
      - `teff` fits specified line by changing temperature, not abundance. Recommended use: use element H and include NLTE for H and Fe
      - `vmic` changes vmic for each abundance line. Very slow, but can get a good vmic. Recommended use: use element Fe
    - `include_molecules` is whether you want molecules in your spectra. Fitting can be faster without them (useful when testing?). Recommended: yes, unless molecules are not expected in the spectra.
    - `nlte` whether want to have NLTE or not. Elements to include with NLTE are written below
    - `fit_vmic`, `fit_vmac` Yes/No/Input depending on if you want to fit them or not. If "no", then microturbulence is calculated based on empirical relation (based on teff, logg, [Fe/H]) and works rather well for FGK-type stars. If Input, it is possible to input microturbulence in the fitlist later. If macroturbulence is "no", then constant one will be applied to all stars (chosen below). If Input, then each star can be given one in the fitlist later on
    - `fit_rotation` Yes/No as well. Yes - fits rotation (not recommended to use together with macroturbulence). No - takes constant one for all below.
    - `element_to_fit` which element to fit. Normally one would fit one element at a time, but it is possible to fit several elements at once using the same linemask (e.g. blended line). If you want to fit abundance for different lines, then you need to fit one element at a time
    - `nlte_elements` which elements to include NLTE for (ignored if `nlte = False`)
    - `linemask_file` is the path in the `linemasks_path` from where the linemask is taken
    - `wavelength_delta` is the synthetic generated `wavelength_delta`. Try not to have it less than observed spectra, but too small will result in slow fitting. Recommended as a start: `0.005`
    - `segment_size` is the size of the generated segment around the line. Recommended as a start: `4`. Not very important, but can be useful to change if nearby lines are very strong and affect the fit (note: H is always generated whether it is in the segment or not)
  - [ExtraParameters]
    - `debug_mode` can be used for debugging code. 0 is best for normal fits, 1 outputs some extra information during the Python fitting, 2 outputs full TS fortran information (a lot of info and much slower fit)
    - `number_of_cpus` is the number of CPUs to use for the fitting. 1 is best for debugging, but can be increased for faster fitting
    - `experimental_parallelisation` parallelises based on each line (not just spectra) for lbl mode. Much faster, but if crashes, then try to set to False (I would recommend to keep it True)
    - `cluster_name` is the name of the cluster, used just for printing. Honestly not very important
  - [InputAndOutputFiles]
    - `input_filename` name of the used fitlist
    - `output_filename` name of the output file (usually `output` and no need to change)
  - [SpectraParameters]
    - `resolution` is resolution of teh spectra. 0 is no convolution based on the resolution
    - `vmac` is default macroturbulence for all stars if `fit_macroturb = No`
    - `rotation` is default macroturbulence for all stars if `fit_rotation = No`
    - `init_guess_elements` are elements to use for initial guess. Only important if you fit several elements at once (e.g. blended line).  Can be several elements: `input_elements_abundance = Mg Ti Ca`
    - `init_guess_elements_path` is the path to the linelist for the initial guess elements. E.g. it can look like this: each line is name of spectra and abundance for the guess [X/Fe]: `HD000001 0.2`. Order of elements should be the same as in `init_guess_elements`
    - `input_elements_abundance` are elements to use for input abundance. This allows to specify abundance of the star for each element. If not specified, then solar scaled abundances are used. Can be several elements: `input_elements_abundance = Mg Ti Ca`
    - `input_elements_abundance_path` is the path to the linelist for the input abundance elements. E.g. it can look like this: each line is name of spectra and abundance [X/Fe]: `HD000001 0.2`. Order of elements should be the same as in `input_elements_abundance`
  - [ParametersForModeAll]
    - `wavelength_minimum` and `wavelength_maximum` specifies the ranges of the fitted spectra
  - [ParametersForModeLbl]
    - `bounds_vmic` are the bounds for microturbulence (HARD BOUNDS)
    - `guess_range_vmic` is the range of microturbulence for the initial guess
    - `find_upper_limit` after the fit is done, it is possible to find upper limit for abundance. This is done by increasing abundance until fitted chi-squared increases by the given `upper_limit_sigma` (e.g. 3 sigma). This is done for each line separately. Doubles the time of the fit, but can be useful to find upper limit or error estimation
  - [ParametersForModeTeff]
    - `bounds_teff` are the bounds for temperature (HARD BOUNDS)
    - `guess_range_teff` is the range of temperature for the initial guess deviated from the input temperature
  - [Bounds]
    - Bounds for vmac, rotation, abundance and doppler shift (deviated from RV)
  - [GuessRanges]
    - Guess ranges for vmac, rotation, abundance and doppler shift (deviated from RV)
- An example of `fitlist` file is added as well:
  - `#name_of_spectrum_to_fit     rv      teff  logg  [Fe/H]  Input_vmicroturb Input_vmacroturb` is first row
    - Most importantly you need specname, rv, teff, logg, [Fe/H] (if fitted) and optionally vmic and vmac
- Now you can run the fitting. To do so, you need to run the `main.py` script. You can do by running:
  - `python3 main.py ./input_files/tsfitpy_input_configuration.cfg` - this will run the fitting on your local computer
  - This will run the job and save output in its unique folder in `./output_files/`
  - `plot_one_star` is a function to plot the results for one star, but it can take extra arguments:
    - `save_figure` is a string, it takes the name of the figure to save WITH the file extension, e.g. `save_figure = 'HD000001.pdf'`. Actual filename will also add line wavelength
    - `xlim` and `ylim` are the limits of the plot
    - `font_size` is the font size of the plot
- You can also easily plot the results:
  - Open `./plotting_tools/plot_output.ipynb` (I would create a copy of it first so that it doesn't interfere with `git pull` later on)
  - Run the first cell to import all the necessary libraries
  - Change the `output_folder_location` to the folder where the output is saved in the second cell
  - Run other cells to plot the results
  - Here you can also generate the synthetic spectra (short window)
    - Look at the last cell: change the folders and change the parameters for the synthetic spectra
    - Just run the cell and it will generate the synthetic spectra
    - You can supply `resolution`, `macro` and `rotation` in `plot_synthetic_data` function to convolve the synthetic spectra
    - You can also supply `verbose=True` to see Fortran output (doesn't work on Mac? Linux only?)

## Usage for grid generation

- 16.06.2023 - finally added grid generation (hooray? please be sure to test before using it maybe? seems to work on my side though)
- To generate the grid, you need to create a `synthetic_spectra_generation_configuration.cfg` file based on the one provided in `./input_files/`
- It is very similar to the config for fitting (just less options) and it only has one extra parameter:
  - `save_unnormalised_spectra` is a boolean, if True, then it will save the unnormalised synthetic spectra (each file would be 30% larger)
- It uses similar `fitlist`, but that one is much more flexible (see example in `./input_files/synthetic_spectra_parameters`)
  - First column specifies columns (order not important). Most importantly to have `teff  logg  [Fe/H]`
  - You can also add options such as `vmic`, `vmac` and `rotation`
  - Of course, you can also add abundance of elements
    - You have to be specific with abundances. `[X/Fe]`, `[X/H]` and `A(X)` (and their combinations) are allowed
    - They are converted to `[X/Fe]` internally basically
    - IMPORTANT!!! Just saying `Mg` would presume that you want `A(Mg)` and not `[Mg/Fe]`
  - Each line is its own spectrum
- After that similarly to before, run `python3 generate_synthetic_spectra.py ./input_files/synthetic_spectra_generation_configuration.cfg`
  - It will generate the grid and save it in the folder specified in the config file
    - Each spectrum is `index.spec` where `index` is the index of the spectrum (same order as in `fitlist`)
  - It will also save the .csv file with the grid parameters (including the name of spectrum)
  - Each spectra also has comments on top with the parameters used to generate it

## Extra notes

Here is the Trello board for the project: https://trello.com/b/2xe7T6qH/tsfitpy-todo

## Some debugging tips:

- If you get a Fortran error `forrtl: severe (24): end-of-file during read, unit -5, file Internal List-Directed Read` in the `bsyn_lu            000000000041DB92  getlele_                   38  getlele.f` just after it trying to `  starting scan of linelist` with some molecular name, then the issue is probably the following one:
  - Your abundance of the element is too low (e.g. I had that issue with [X/Fe] = -30) and it skips that element in the molecular line identification. In that case remove the molecular linelist containing that element, or increase your element abundance (e.g. to [X/Fe] = -3)