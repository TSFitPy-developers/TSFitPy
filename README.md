# Turbospectrum Spectral Fitting with Python (TSFitPy)

TSFitPy is a pipeline designed to determine stellar abundances and atmospheric parameters through the use of Nelder-Mead (simplex algorithm) minimization. It calculates model spectra "on the fly" while fitting instead of using a more sophisticated method that relies on training neural networks. Using this method allows the pipeline to gain flexibility in the stellar physics used at the expense of computation time, which makes it useful for small datasets.

To use TSFitPy, you will need a working Turbospectrum (TS) installation of the latest version, which has the capability to compute LTE or NLTE spectra. TSFitPy has not been tested on older versions of TS. The latest version of TS can be found here: https://github.com/bertrandplez/Turbospectrum_NLTE

The code requires at least version Python 3.7. It also makes use of fortran programs, which will need to be compiled on the user's machine (intel fortran ifort/ifx compiler highly recommended, but works with gfortran as well). The Python packages needed are as follows (they should all be installable via "pip install"):
- Numpy
- Scipy (at least 1.7.1)
- Dask (installed using `pip install dask[complete]`)
- dask-jobqueue (does NOT come by default with the Dask)
- Pandas
- Astropy
- Matplotlib

Also, Windows is not supported (?).

There is a WIP (developed by NS only atm) GUI for TSFitPy (at least results plotting). You can see whether you might like it. It is available [here](https://github.com/stormnick/TSGuiPy).

---

*When you don’t read the docs, every bug is a feature and every feature is a bug.*

## Table of Contents
0. [Acknowledgements](#acknowledgements)
1. [Quick Start](#quick-start)
2. [Extra installation info](#extra-installation-info)
3. [Usage](#usage)
   - [Fitting Spectra](#fitting-spectra)
   - [Generating Synthetic Spectra](#generating-synthetic-spectra)
   - [Calculating NLTE Corrections](#calculating-nlte-corrections)
   - [NLTE Usage](#nlte-usage)
4. [Plotting Tools](#plotting-tools)
5. [Compiler Notes: ifort / ifx](#compiler-notes-ifort-ifx)
6. [Parallelisation and Clusters](#parallelisation-and-clusters)
7. [Troubleshooting & Flags](#troubleshooting--flags)
8. [Conclusion](#conclusion)
9. [FAQ](#faq)

---

## Acknowledgements

If you use this code, please acknowledge the authors of the code and the Turbospectrum code. **The most important papers to reference are:**

- TS NLTE + TSFitPy [Gerber, J. M. et al. 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...669A..43G/abstract) and [Storm, N. & Bergemann M. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3718S/abstract)

Please reference these two papers in your work if you use TSFitPy, as they describe the code and its capabilities, and contain references to most other relevant papers. However, if you want to reference specifics, then here is the full list of papers to acknowledge, depending on what you use in TSFitPy:

- Original TS [Alvarez, R & Plez, B. 1998](https://ui.adsabs.harvard.edu/abs/1998A%26A...330.1109A/abstract)
- TS [Plez, B. 2012](https://ui.adsabs.harvard.edu/abs/2012ascl.soft05004P/abstract)
- TS NLTE + TSFitPy [Gerber, J. M. et al. 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...669A..43G/abstract) and [Storm, N. & Bergemann M. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3718S/abstract)
- Solar abundances input [Bergemann, M. et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2236B/abstract) and [Magg, E. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.140M/abstract)
- 1D MARCS models [Gustafsson, B. et al. 2008](https://ui.adsabs.harvard.edu/abs/2008A%26A...486..951G/abstract)
- <3D> STAGGER models (if using those) [Magic, Z. et al. 2013](https://ui.adsabs.harvard.edu/abs/2013A%26A...557A..26M/abstract)
- Gaia-ESO linelist (if using that one) [Heiter, U. et al. 2021](https://ui.adsabs.harvard.edu/abs/2021A%26A...645A.106H/abstract)
  - With new atomic data for C, N, O, Si, Mg, as described in [Magg, E. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.140M/abstract)
  - Gaps of it are filled with VALD (if using those) [Ryabchikova, T. et al. 2015](https://ui.adsabs.harvard.edu/abs/2015PhyS...90e4005R/abstract)

If you make use of the `teff` or `vmic` fitting methods, please acknowledge the following paper with the description of the method:

- D'Orazi, V. et al. [D'Orazi, V. et al. 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.tmp.1155D/abstract)

If you make use of the NLTE data, please acknowledge the appropriate papers for the NLTE data used (different one for each element!):

- H: [Mashonkina, L. et al. 2008](https://ui.adsabs.harvard.edu/abs/2008A%26A...478..529M/abstract)
- O: [Bergemann, M. et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2236B/abstract)
- Na (qmh): [Ezzeddine, R. et al. 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A.141E/abstract)
- Mg: [Bergemann, M. et al. 2017](https://ui.adsabs.harvard.edu/abs/2017ApJ...847...15B/abstract)
- Al (qmh): [Ezzeddine, R. et al. 2018](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A.141E/abstract)
- Si: [Bergemann, M. et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...764..115B/abstract) and [Magg, E. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022A%26A...661A.140M/abstract)
- Ca: [Mashonkina, L. et al. 2017](https://ui.adsabs.harvard.edu/abs/2017A%26A...605A..53M/abstract) and [Semenova, E. et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...643A.164S/abstract)
- Ti: [Bergemann, M. et al. 2011](https://ui.adsabs.harvard.edu/abs/2011MNRAS.413.2184B/abstract)
- Cr: [Bergemann, M. & Cescutti, G. 2010](https://ui.adsabs.harvard.edu/abs/2010A%26A...522A...9B/abstract)
- Mn: [Bergemann, M. et al. 2019](https://ui.adsabs.harvard.edu/abs/2019A%26A...631A..80B/abstract)
- Fe: [Bergemann, M. et al. 2012b](https://ui.adsabs.harvard.edu/abs/2012MNRAS.427...27B/abstract) and [Semenova, E. et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...643A.164S/abstract)
- Co: [Bergemann, M. et al. 2010](https://ui.adsabs.harvard.edu/abs/2010MNRAS.401.1334B/abstract) and [Yakovleva, S.A. et al. 2020](https://www.mdpi.com/2218-2004/8/3/34)
- Ni: [Bergemann, M. et al. 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.508.2236B/abstract) and [Voronov, Y.V. et al. 2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...926..173V/abstract)
- Sr: [Bergemann, M. et al. 2012a](https://ui.adsabs.harvard.edu/abs/2012A%26A...546A..90B/abstract) and [Gerber, J. M. et al. 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...669A..43G/abstract)
- Y: [Storm, N. & Bergemann M. 2023](https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.3718S/abstract) and [Storm, N. et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...683A.200S)
- Ba: [Gallagher, A. et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..55G/abstract)
- Eu: [Storm, N. et al. 2024](https://ui.adsabs.harvard.edu/abs/2024A%26A...683A.200S)

## Quick Start
A rapid guide to confirm that TSFitPy works, please see [extra notes](#extra-installation-info) for all linelists, model atmospheres, and NLTE data.

1. **Clone TSFitPy**:
   ```bash
   git clone https://github.com/TSFitPy-developers/TSFitPy.git
   cd TSFitPy/turbospectrum/
   rm readme.txt
   cd ..
   ```
2. **Get TurboSpectrum**:
   ```bash
   git clone https://github.com/bertrandplez/Turbospectrum_NLTE.git turbospectrum
   ```
3. **Compile TurboSpectrum**:
   ```bash
   cd turbospectrum/exec-intel/
   # If using gnu: cd turbospectrum/exec-gf/
   # If using ifx: cd turbospectrum/exec-ifx/
   # On Linux, consider uncommenting 'mcmodel=medium' on line 11 of the Makefile.
   make
   cd ../../
   ```
4. **Download MARCS model atmospheres**:
   - [1D MARCS standard composition models](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/)
   - Place the unzipped `.mod` files in `TSFitPy/input_files/model_atmospheres/1D/`
5. **Download sample linelist**:
   - [NLTE Gaia-ESO Linelist](https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/), file `nlte_ges_linelist_jmgXX20XX_I_II` for atomic lines between 4200-9200 Å
   - Put the relevant file(s) in `TSFitPy/input_files/linelists/linelist_for_fitting/`
6. **Copy sample spectrum**:
   ```bash
   cp -r input_files/sample_spectrum/* input_files/observed_spectra/
   ```
7. **Copy TurboSpectrum interpolators**:
   ```bash
   cp -r turbospectrum/interpolator/* scripts/model_interpolators/
   ```
8. **Compile the model interpolators**:
   ```bash
   cd scripts/
   python3 compile_fortran_codes.py
   # Choose 'GNU', 'IFORT', or 'IFX' when prompted
   cd ..
   ```
9. **Run a test fit**:
   ```bash
   python3 main.py ./input_files/tsfitpy_input_configuration.cfg
   ```
   Example successful output:
   ```
   [...]
   [Fe/H]=-0.3180 rv= 0.3268 vmic= 1.0516 ... Converged: Fe: -0.32 Number of iterations: 14
   [...]
   [Fe/H]= 0.0282  rv= 0.2282 vmic= 1.0679 ... Converged: Fe: 0.03 Number of iterations: 10
   TSFitPy had normal termination
   Fitting completed
   End of the fitting: XXX-XX-20XX-XX-XX-XX
   ```
10. **Optional: Plot results**:
   ```bash
   cd plotting_tools/
   cp plot_output.ipynb plot_output_test.ipynb
   # Run the first cell for imports
   # Modify 'output_folder_location' in plot_output_test.ipynb to your output folder.
   # E.g. you will find a folder `TSFitPy/output_files/XXX-XX-20XX-XX-XX-XX_0.XXXXXXXXXXXXXXXX_LTE_Fe_1D/`, so copy `XXX-XX-20XX-XX-XX-XX_0.XXXXXXXXXXXXXXXX_LTE_Fe_1D` and paste it instead of `OUTPUTCHANGEHERE`
   # Run the next few notebook cells to produce your plots.
   ```
![Fitted image](./example_image_plot/fitted_star_example.png)

> **If everything ran without errors, your installation works.** Proceed to the full instructions below for in-depth usage.

---

## Extra installation info

1. **Linelists**. Place all of the desired ones in `TSFitPy/input_files/linelists/linelist_for_fitting/`.
   - **Important**: Do **not** combine overlapping line data for the same wavelength ranges **because all of the linelists in the folder will be used**.
     - You do not want to synthesize the same line twice.
   - You need to have linelists in TS format [see TS docs](https://github.com/bertrandplez/Turbospectrum_NLTE/blob/master/DOC/Turbospectrum_v20_Documentation_v6.pdf)
     - You can convert from VALD format using [TS fortran script](https://github.com/bertrandplez/Turbospectrum_NLTE/blob/master/Utilities/vald3turbo.f)
     - For NLTE you will need to convert a new linelist using the `convert_lte_to_nlte.py` script in `TSFitPy/scripts/` folder (see [NLTE Usage](#nlte-usage) for more details)
   - Individual linelists (recommended as a start):
     - **Gaia-ESO NLTE + VALD NLTE linelists**: [link](https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/)
       - `nlte_ges_linelist_jmgDATE_I_II`
       - `vald-3700-3800-for-grid-nlte-DATE`
       - `vald-3800-4200-for-grid-nlte-DATE`
       - `vald-9200-9300-for-grid-nlte-DATE`
       - `vald-9300-9800-for-grid-nlte-DATE`
     - **Molecular Linelists**: [link](https://keeper.mpdl.mpg.de/d/3a5749b0bb5d4e0d8f4f/), folder `molecules-420-920nm`. 
       - Download AND **unzip** each individual file and place them in `TSFitPy/input_files/linelists/linelist_for_fitting/`
     - **H lines** should be included by default in the file `Hlinedata`
2. **Model Atmospheres**: 1D and (optional) <3D> STAGGER models are available.
   - [1D MARCS Models](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) → `TSFitPy/input_files/model_atmospheres/1D/`
   - [3D STAGGER Models](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) → `TSFitPy/input_files/model_atmospheres/3D/`
   - Unzip them so `.mod` files sit in their respective folders.
3. **(Optional) NLTE Departure Coefficients**. There are two options: manually download the NLTE grids or use the `/utilities/download_nlte_grids.py` python script.
    - **Automatic Download**:
      - Go to `TSFitPy/utilities/` and run:
        ```bash
        python3 download_nlte_grids.py PATH_WHERE_TO_SAVE 1D/3D ELEMENT1 ELEMENT2 ...
        ```
        - Example: `python3 download_nlte_grids.py ./input_files/nlte_data/ 1D Ba Sr` will download NLTE grids for Ba and Sr in 1D MARCS models.
        - You can specify multiple elements, e.g. `Ba Sr Y`.
        - If you want to download 3D grids, use `3D` instead of `1D`. Use `1D,3D` or `all` to download both.
        - If you want to download all elements, use `all` instead of `ELEMENT1 ELEMENT2 ...`.
        - The script will NOT overwrite existing files. So if something fails, you can just run it again. Just be sure to delete half-downloaded or corrupted files, as they will not be overwritten.
      - Syntax to download EVERYTHING (warning: this will take ~600 GB of space):
        ```bash
        python3 download_nlte_grids.py ../input_files/nlte_data/ all all
        ```
    - **Manual Download**:
      - Download from [NLTE dep grids](https://keeper.mpdl.mpg.de/d/6eaecbf95b88448f98a4/) (folder `dep grids`) → `TSFitPy/input_files/nlte_data/`
      - Place (unzipped) `.bin` and `auxData_...` in dedicated element folders (e.g. `nlte_data/Ba/`) and put `atom.ELEMENT_AND_NUMBER` in `nlte_data/model_atoms/`.
      - All the NLTE grids for all elements take roughly 1 TB total. However, usually you only need very few elements, so except to requite around 20-100 GB of space, depending on usage.
4. **Line Masks** (i.e., wavelength ranges to fit)
   - Located in `TSFitPy/input_files/linemask_files/`
   - Each folder typically has one example file: `./ELEMENT/ELEMENT-lmask.txt`
     - Doesn't matter how you call them, but the format is important. Feel free to create your own, example linemasks are not the best ones out there.
   - Format: `[center_wavelength] [left_wing] [right_wing]`. Use `;` for comments. Wings might change significantly depending on your resolution. Example:
```
5000 4999.5 5000.5
5234 5233.5 5234.5 ; good line
; 6000 5999.5 6000.5 ; this line is not used
```

---

## Usage
TSFitPy is designed for flexible stellar spectral fitting, but it does require some knowledge of what you are fitting and how. Common tasks include:

1. [Fitting Spectra](#fitting-spectra)
2. [Generating Synthetic Spectra](#generating-synthetic-spectra)
3. [Calculating NLTE Corrections](#calculating-nlte-corrections)

First let's go through the commonly used files.

### Configuration Files
TSFitPy uses a `.cfg` file to specify:
- Compiler options
- Paths
- Fitting mode
- Atmosphere type
- NLTE usage, line masks, and other advanced flags

**We recommend copying the default config (`tsfitpy_input_configuration.cfg`) and renaming it for each run**, so any future code updates won’t overwrite your settings. More information one each config setting can be found in the [Configuration File Options](#configuration-file-options) section.

### Fitlist Files
TSFitPy reads star or synthesis parameters from a “fitlist” file:
```text
name_of_spectrum rv  teff    logg    [Fe/H]  vmic  vmac  rotation  resolution  ...
HD000001         10  5000.0  2.0     0.0     1.0   1.0   0.0       80000
HD000002         -5  6000.0  4.0     -4.0    1.5   3.0   1.0       70000
```
- **First (always) column** is the file name of the observed spectrum.
  - It is skipped and not needed in the [Generating Synthetic Spectra](#generating-synthetic-spectra) script.
- **Remaining columns** can appear in any order, but the headers must match the parameter names (e.g. `rv`, `teff`, `logg`, `[Fe/H]`, `vmic`, `vmac`, `[X/Fe]`, `rotation`, `resolution`, `snr`).
  - `rv` is the radial velocity of the star (in km/s)
  - `teff` is the effective temperature of the star (in K)
  - `logg` is the surface gravity of the star (in cgs)
  - `[Fe/H]` is the metallicity of the star (in dex)
  - `Input_vmicroturb` (or `vmic`) is the microturbulence of the star (in km/s)
  - `Input_vmacroturb` (or `vmac`) is the macroturbulence of the star (in km/s)
  - `rotation` for the rotation of the star (in km/s)
  - `resolution` is the resolution of the star (R value)
  - `snr` is the signal to noise ratio of the star (if not provided, it will be set to 100 by default), used only for chi2 calculation
    - Which will be used to estimate an error using formula `sigma = 1/snr`, if no error is provided in the spectra
  - `[X/Fe]`, `[X/H]`, `A(X)` are the abundances of the star (in dex), relative to Fe, H or absolute. You can use any X value and however many you want
    - **IF YOU DO NOT PROVIDE** value for any elements, then they ALL (including alpha) will be [Fe/H] scaled solar abundances
- If no macroturbulence and rotation is provided, TSFitPy can fit or set them to zero.
- If no microturbulence is provided, then it can be fitted, or be set according to teff/logg/feh value (see [How the function calculates it here](https://github.com/TSFitPy-developers/TSFitPy/blob/v2.0.2/scripts/auxiliary_functions.py#L24)).
- Errors: If you supply a third column in your actual observed spectrum file, TSFitPy interprets it as flux error (σ). If not present, default error is 0.01 or 1/SNR. Please be careful NOT to supply e.g. unnormalised fluxes in the third column, as TSFitPy will interpret it as error and will not fit it correctly.

---

### Fitting Spectra

The steps to fit a spectrum are as follows:

1. **Normalise spectra** (outside TSFitPy). TSFitPy expects already perfectly-normalised spectra in `TSFitPy/input_files/observed_spectra/`. 
   - There is no "normalisation-fitting/tweaking" in TSFitPy. Any normalisation issues will affect the fit.
2. **Choose or create a `fitlist`** that lists your star(s) with relevant initial parameters (e.g., `RV`, `Teff`, `[Fe/H]`, etc.). See [Fitlist Files](#fitlist-files) for details.
3. **Copy and edit the config file** (e.g., `tsfitpy_input_configuration.cfg`) to specify atmosphere type (`1D`/`3D`), `fitting_mode` (`lbl`, `all`, `vmic`, etc.), line mask file, NLTE usage, etc.
   - You should create a separate config for each run, so you can keep track of different settings.
   - More information on each config setting can be found in the [Configuration File Options](#configuration-file-options) section.
4. **Run**: `python3 main.py ./input_files/tsfitpy_input_configuration_YOUR_NAME.cfg`
5. **Output** is automatically placed in `./output_files/DATE_TIME_.../`.
   - Contains an output of fitted parameters and flags for each line.
   - The code sets `flag_error` and `flag_warning` bits if potential issues appear.
6. Look at the fits (recommended: by eye; at least for a few of them). Remove any fits that are bad (either by eye or by chi2, `flag_error`, `flag_warning`).
7. **Getting an average abundance**. Use `./plotting_tools/analyse_the_output.py` with arguments:
   ```bash
   python3 analyse_the_output.py ../output_files/OUTPUT_FOLDER_NAME/ --remove-errors --remove-warnings --chisqr-limit 5 --ew-limits 1 200 --ew-limit-total 350
   ```
   - This will create a file with average abundances for each element.
   - Arguments (if not given, defaults are used):
     - File path to the output folder (e.g., `../output_files/XXX-XX-20XX-XX-XX-XX_0.XXXXXXXXXXXXXXXX_LTE_Fe_1D/`)
     - `--remove-errors` or `--no-remove-errors`: remove lines with `flag_error != 0` (recommended)
     - `--remove-warnings` or `--no-remove-warnings`: remove lines with `flag_warning != 0` (recommended)
     - `--chisqr-limit`: remove lines with `chi_squared > X` (default: 5). Increase for lower SNR and bigger linemasks.
     - `--ew-limits`: remove lines with `ew < X` or `ew > Y` (excluding blends) (default: 1 and 200, respectively). Increase for molecular lines or bigger linemasks.
     - `--ew-limit-total`: remove lines with `ew + ew_blend > X` (default: 350). Increase for molecular lines or bigger linemasks.
   - If you use TSGuiPy and download `new_flags.csv` file, you can use put it into the folder with the output and instead run:
   ```bash
    python3 analyse_the_output_new_flags.py ../output_files/OUTPUT_FOLDER_NAME/ --remove-errors --no-remove-warnings --chisqr-limit 50 --ew-limits 1 400 --ew-limit-total 550
    ```

Examples of a fitlist were given before, but let's quickly go through the example of an output.
```text
specname	wave_center	wave_start	wave_end	Doppler_Shift_add_to_RV	Fe_H	Ca_Fe	Microturb	Macroturb	rotation	chi_squared	        ew	        flag_error	flag_warning	ew_just_line	ew_blend	ew_sensitivity
15PEG	        5188.844	5188.350        5189.250        0.101	                -0.6438	0.0121	1.6622	        0.0	        11.1371         2.2409336243956903	154.8608        00000000	10000000        57.37060        97.49020        61.8117
15PEG	        5260.387	5260.018        5265.738        0.201	                -0.6438	0.0691	1.6622	        0.0	        11.1371         3.9124957259256727	471.4730        00000100	10000000        215.2479        256.2251        246.9036
31AQL	        5188.844	5188.350        5189.250        0.301	                0.2001	0.0441	1.2679	        0.0	        6.92520         12.007318861323174	266.4543        00000000	10000000        120.4375        146.0168        65.5899
31AQL	        5260.387	5260.018        5265.738        0.401	                0.2001	-0.0126	1.2679	        0.0	        6.92520         20.359021312974495	1340.178        00000100	10000000        426.9957        913.1823        257.481
```
- `specname`: name of the observed spectrum
- `wave_center`: center of the line (the one you used in the linemask file). Technically not used in the fit, it is mostly for your reference
- `wave_start`: left wing of the line (used in the chi2 calculation)
- `wave_end`: right wing of the line (used in the chi2 calculation)
- `Doppler_Shift_add_to_RV`: extra RV added to the RV of the spectrum (so total RV for the line = `Doppler_Shift_add_to_RV` + `RV` from the fitlist)
- `Fe_H`: [Fe/H] from the fitlist or the one that was fitted (if fitting [Fe/H])
- `Ca_Fe`: here [Ca/Fe] was fitted, but it can be any element. Not given if not fitted.
- `Microturb`: microturbulence from the fitlist or the one that was fitted
- `Macroturb`: macroturbulence from the fitlist or the one that was fitted
- `rotation`: rotation from the fitlist or the one that was fitted
- `chi_squared`: reduced chi2 of the fit (divided by the number of points; 1.0 is a good fit, <1.0 is a too good fit, >1.0 is a bad fit). The scaling will depend on the SNR/error provided of the spectrum. If not provided, the default is 0.01 or 1/SNR. So do not overinterpret the chi2 value if you did not provide the error.
- `ew`: equivalent width of the line (in mÅ) between `wave_start` and `wave_end` (including the blended lines)
- `flag_error`: bit flag for errors (see [Troubleshooting & Flags](#troubleshooting--flags) for details)
- `flag_warning`: bit flag for warnings (see [Troubleshooting & Flags](#troubleshooting--flags) for details)
- `ew_just_line`: equivalent width of the line (in mÅ) between `wave_start` and `wave_end` (excluding the blended lines)
- `ew_blend`: equivalent width of the blended lines (in mÅ) between `wave_start` and `wave_end`
- `ew_sensitivity`: by how much EW changes if you change the abundance of the element (units are mÅ/dex). This is a good indicator of how sensitive the line is to the abundance. If it is low, it means that the line is not very sensitive to the abundance and you should be careful with the abundance you get from it. If it is high, then you can trust the abundance more.

In our example, we have two stars with 2 lines each. In this case I recommend to look through the fits by eye. In case of many stars (hundreds-thousands), you can try to get an average of all abundances by removing any bad fits (e.g. chi2 > 10, flag_error != 0, ew_sensitivity > 1, etc.). Then running your own script, or using my example script in `plotting_tools/analyse_the_output.py` to get the average abundances.

---

### Generating Synthetic Spectra
**Use `generate_synthetic_spectra.py`** with a separate config (based on `synthetic_spectra_generation_configuration.cfg`). The process is:
1. **Create a synthetic “fitlist”** specifying the grid of parameters for which to generate spectra (e.g. `teff`, `logg`, `[Fe/H]`, `[X/Fe]`, `vmic`, `vmac`, `rotation`; last 4 are optional, of course).
2. **Edit config** to define paths, atmosphere type, wavelengths, etc. Optional unnormalised spectra can also be saved using an appropriate flag in the config.
3. **Run**: `python3 generate_synthetic_spectra.py ./input_files/synthetic_spectra_generation_configuration.cfg`
4. **Output**: each synthetic spectrum is saved as `index.spec`, with a companion `.csv` containing the parameter grid. 
   - Please check the spectra carefully. Some spectra will have NaNs or negative fluxes due to NLTE convergence issues. When processing, feel free to remove any such spectra using `if np.any(np.isnan(spectrum)) or np.any(spectrum <= 0): continue` in the code.

Example of a fitlist to generate spectra:
```text
Teff logg Fe/H vmic C/Fe Ca/Fe Ba/Fe O/Fe
5403.27 2.342 -1.494 1.541 -0.537 -0.258 -1.19 -0.258
4125.85 4.895 -3.899 2.853 -0.182 0.81 -1.183 0.814
```

Example of the `spectra_parameters.csv`:
```text
specname,rv,teff,logg,feh,vmic,C_Fe,Ca_Fe,Ba_Fe,O_Fe
0.spec,0.0,5403.27,2.342,-1.494,1.541,-0.537,-0.258,-1.19,-0.258
```
As you can see, it reflects the parameters of the fitlist. Also, you might notice that `1.spec` is missing. That is because TS failed to generate it, so the file `1.spec` doesn't exist and nor does the corresponding `1.csv` file. Let's take a look at the `0.spec`:
```text
#Generated using TurboSpectrum and TSFitPy wrapper
#date: 2025-03-03 10:04:18.787979
#spectrum_name: 0.spec
#teff: 5403.27
#logg: 2.342
#[Fe/H]: -1.494
#vmic: 1.541
#vmac: 0.0
#resolution: 0.0
#rotation: 0.0
#nlte_flag: True
#[C/Fe]=-0.5370 LTE
#[Ca/Fe]=-0.2580 NLTE
#[Ba/Fe]=-1.1900 NLTE
#[O/Fe]=-0.2580 NLTE
#[Fe/Fe]=0.0 (solar scaled) NLTE
#
#Wavelength Normalised_flux Unnormalised_flux
6110.0  0.99998  5575560.0
6110.005  0.99998  5575550.0
[...]
```
- The first lines are comments with the parameters used to generate the spectrum. So you can easily check the parameters used.
  - It also writes which elements were done in NLTE. Every other element is LTE and solar-scaled.
- The first column is the wavelength in Å, the second column is the normalised flux, and the third column is the unnormalised flux. The last one is optional and can be removed if you don't need it.


---

### Calculating NLTE Corrections
**Use `calculate_nlte_correction_line.py`** to compute the NLTE correction for a single line. The correction is done by finding by how much abundance changes for NLTE spectra, to match NLTE EW to the LTE EW. The LTE EW is calculated based on the input spectra and abundance ([X/Fe] = 0, if not specified).
1. **Create a fitlist** with the desired parameters for the NLTE correction. The first column should be the name of the model atmosphere; that will be saved in the output (e.g., `4000_1.0_-5.0.spec`). You can (but don't have to) specify the LTE abundance of the element as well for the correction.
2. **Specify** the target element, line mask, and atmospheric parameters in the config (it uses the same config structure as `tsfitpy_input_configuration.cfg`).
3. **Run** the script:
   ```bash
   python3 calculate_nlte_correction_line.py ./input_files/tsfitpy_input_configuration.cfg
   ```
4. **Result** is saved in the output folder, detailing the difference between LTE and NLTE for that specific transition.

Example of a fitlist to get NLTE corrections for a Mg line for three model atmospheres (last column is optional; assumed as 0 otherwise):
```text
#name              rv  teff  logg [Fe/H] vmic [Mg/Fe]
4000_1.0_-5.0.spec 0.0 4000   1.0 -5.0    2.0    0.4
4000_1.0_-4.0.spec 0.0 4000   1.0 -4.0    2.0    0.3
4000_1.0_-3.0.spec 0.0 4000   1.0 -3.0    2.0    0.2
```

This is how to setup the config file for NLTE correction:
```text
[FittingParameters]
atmosphere_type = 1D                    # change to 3D if you want to get 1D to 3D corrections
fitting_mode = lbl                      # keep as lbl
include_molecules = True                # doesn't matter
nlte = True                             # if you want to get NLTE corrections
fit_vmic = Input                        # to take vmic from the input file. Set to `no` to calculate based on teff/logg/vmic
fit_vmac = No                           # doesn't matter
fit_rotation = No                       # doesn't matter
element_to_fit = Eu                     # element to correct
nlte_elements =  Eu                     # element to correct
linemask_file = Eu/eu-lmask_corr.txt    # linemask specifies which line(s) to correct, same format as linemask files
wavelength_delta = 0.001                # set to low enough for a more accurate correction, <= 0.01
segment_size = 5                        # doesn't matter
```

As the output you will get something like this:
```text
#specname	        teff    logg	met	    microturb	wave_center	ew_lte	                ew_nlte	                nlte_abund	        nlte_correction
4000_1.0_-5.0.spec	4000    1.0     -5.0        2.0	        16718.957	313.9284500000539	313.92865000001535	-0.7504994999989965	-0.21439039999899645
4000_1.0_-4.0.spec	4000    1.0     -4.0        2.0	        16750.539	426.4978999999922	426.4979000000118	-0.7846824999993689	-0.2485733999993689
4000_1.0_-3.0.spec	4000    1.0     -3.0        2.0	        16763.359	168.93209999992285	168.9320999999353	-0.6534184999988184	-0.1173093999988184
```
- `specname`: name of the model atmosphere
- `teff`, `logg`, `met`, `microturb`: parameters of the model atmosphere
- `wave_center`: center of the line (the one you used in the linemask file)
- `ew_lte`: LTE equivalent width
- `ew_nlte`: NLTE equivalent width; it should be the same as the LTE equivalent width. If it is not - something is wrong with the NLTE correction; don't use it
- `nlte_abund`: NLTE abundance of the element (i.e., the abundance that you need to use to get the same EW as LTE)
- `nlte_correction`: NLTE correction of the element (i.e., the difference between LTE and NLTE abundance). This is the value you need to use to correct the LTE abundance to NLTE abundance.

---

### NLTE Usage
For NLTE calculations, you need:
- A **valid departure coefficient grid** for the element (e.g., Ba, Fe, O, Mg, etc.). Place these in `nlte_data/ElementName/`.
- A **model atom** in `nlte_data/model_atoms/` (e.g., `atom.ba111`, `atom.fe607a`, etc.). This is the model atom used for NLTE calculations.
- A **linelist** containing the correct transition-level mappings to your chosen model atom. You’ll see lines tagged with something like: `6 26 '3s5S2*' '4p5P3'` referencing the atomic levels at the end of each line for the appropriate element.
- The config file must set `nlte = True` and list the relevant `nlte_elements`.

> **Important**: If lines in the linelist aren’t matched to model atom levels, they revert to LTE. Use the script `utilities/convert_lte_to_nlte.py` to help label lines if needed.

---

### Configuration File Options

- The config file should already be ready for a test run, but here is the reference breakdown if needed
  - [turbospectrum_compiler]:
    - `compiler` specifies the compiler (`ifort`, `ifx` or `gnu`). Location of turbospectrum is expected at `TSFitPy/turbospectrum/`
  - [MainPaths]
    - Next few lines specify the paths. Default paths are relative to the `TSFitPy/scripts/TSFitPy.py`, but it is possible to change paths if you want to keep your data in a separate folder (e.g. it can be useful if sharing data on a cluster)
  - [FittingParameters]
    - `atmosphere_type`: `1D` or `3D`. MARCS is 1D, STAGGER average are <3D> models
      - Note that "<3D> models" have depth-dependent microturbulence (vmic). So inputting ANY vmic will have NO effect on the spectra for <3D> model based spectra
    - `fitting_mode` specifies fitting mode
      - `all` fits all lines within the linemask at the same time. Advantage: faster. Disadvantage: cannot know whether any specific line has good or bad fit. **Not recommended**
      - `lbl` fits all lines within the linemask one line at a time. Advantage: get full info for each line with separate abundance, macroturbulence etc. Can also fit microturbulence (not very well though?) Disadvantage: slower
      - `teff` fits specified line by changing temperature, not abundance. Recommended use: use element H and include NLTE for H. Also recommend to mask out cores of H-lines in your spectra
      - `vmic` changes vmic for each abundance line. Very slow, but can get a better vmic estimate. Recommended use: use element Fe and fit both Fe and vmic together
      - `logg` fits specified line by changing logg, not abundance. **NOT TESTED WELL, but I don't see why it wouldn't work. Use with caution**
    - `include_molecules` is whether you want molecules in your spectra. Fitting can be faster without them (useful when testing?). Recommended: yes, unless molecules are not expected in the spectra.
    - `nlte`: `True`/`False`: whether want to have NLTE or not. Elements to include with NLTE are written below
    - `fit_vmic`: `Yes`/`No`/`Input`. `No` - microturbulence is calculated based on empirical relation (based on teff, logg, [Fe/H]) and works rather well for FGK-type stars (see `TSFitPy/scripts/auxiliary_functions.py:calculate_vturb()`). `Input` - microturbulence is taken from the fitlist (**Recommended** unless you cannot fit vmic/don't know vmic). `Yes` - microturbulence is fitted (**NOT recommended**). Use 'vmic' fitting mode to fit vmic instead of using this option
    - `fit_vmac`, `fit_rotation`: `Yes`/`No`/`Input`. `No` - macroturbulence/rotation are set to 0. `Input` - macroturbulence/rotation is taken from the fitlist. `Yes` - macroturbulence/rotation is fitted. Recommended to either fit or take from input. Also doing only one is usually fine, unless you have a fast rotator.
    - `element_to_fit` which element to fit. Normally one would fit one element at a time, but it is possible to fit several elements at once using the same linemask (e.g. blended line) (**NOT recommended**). If you want to fit abundance for different lines, then you need to fit one element at a time
      - **IMPORTANT**: Providing several elements, will fit several elements for all lines within THE SAME linemask. So if you want to fit several elements for different lines, you need to create separate config files for each element
    - `nlte_elements` which elements to include NLTE for (ignored if `nlte = False`)
    - `linemask_file` is the path in the `linemasks_path` from where the linemask is taken
    - `wavelength_delta` is the synthetic generated `wavelength_delta`. Try not to have it less than observed spectra, but too small will result in slow fitting.
    - `segment_size` is the size of the generated segment around the line. Recommended as a start: `4`. Not very important, but can be useful to change if nearby lines are very strong and affect the fit (note: H is always generated whether it is in the segment or not)
  - [ExtraParameters]
    - `debug_mode` can be used for debugging code. 0 is best for normal fits, 1 outputs some extra information during the Python fitting, 2 outputs full TS fortran information (a lot of info and much slower fit). -1 will minimise the output to almost nothing.
    - `number_of_cpus` is the number of CPUs to use for the fitting (multiprocessing). 1 is best for debugging, but can be increased for faster fitting. Maximum: number of cores on your machine
    - `experimental_parallelisation` currently does NOTHING (experimental is now part of the main code). Might do something later, who knows...
    - `cluster_name` is the name of the cluster, used just for printing. Honestly not very important
  - [InputAndOutputFiles]
    - `input_filename` name of the used fitlist file
    - `output_filename` name of the output file (usually `output` and no need to change)
  - [SpectraParameters]
    - `resolution` is resolution of the spectra (big R). 0 is no convolution based on the resolution. Usually your R will be around 10000-100000 (it is NOT FWHM)
    - `vmac` is the default macroturbulence for all stars if `fit_macroturb = No` (alternatively can put individual vmac for each star in the fitlist file)
    - `rotation` is the default macroturbulence for all stars if `fit_rotation = No` (alternatively can put individual rotation for each star in the fitlist file)
    - Old and slightly outdated (but still supported) parameters:
      - `init_guess_elements` are elements to use for initial guess. Only important if you fit several elements at once (e.g. blended line).  Can be several elements: `input_elements_abundance = Mg Ti Ca`
      - `init_guess_elements_path` is the path to the linelist for the initial guess elements. E.g. it can look like this: each line is name of spectra and abundance for the guess [X/Fe]: `HD000001 0.2`. Order of elements should be the same as in `init_guess_elements`
      - `input_elements_abundance` are elements to use for input abundance. This allows to specify abundance of the star for each element. If not specified, then solar scaled abundances are used. Can be several elements: `input_elements_abundance = Mg Ti Ca`
      - `input_elements_abundance_path` is the path to the linelist for the input abundance elements. E.g. it can look like this: each line is name of spectra and abundance [X/Fe]: `HD000001 0.2`. Order of elements should be the same as in `input_elements_abundance`
  - [ParametersForModeAll]
    - `wavelength_minimum` and `wavelength_maximum` specifies the ranges of the fitted spectra **ONLY FOR THE `all` fitting mode. Normally it is not needed to change this**
  - [ParametersForModeLbl]
    - `bounds_vmic` are the bounds for microturbulence (HARD BOUNDS)
    - `guess_range_vmic` is the range of microturbulence for the initial guess
    - `find_upper_limit` after the fit is done, it is possible to find upper limit for abundance. This is done by increasing abundance until fitted chi-squared increases by the given `upper_limit_sigma` (e.g. 3 sigma). This is done for each line separately. Doubles the time of the fit, but can be useful to find upper limit or error estimation
    - `fit_continuum` True/False, whether to linearly adjust the local contrinuum around the line. Equation is flux_norm += 1 - ((wavelength - wavelength_left_linemask_line) * slope + intercept)
    - `bounds_continuum_slope` slope in the equation above. Slope = 0, means just a flat up/down shift
    - `bounds_continuum_intercept` intercept of the continuum. Intercept = 1, means that the continuum is at 1. Slope 0 and interecept 1 means that continuum was not adjusted
  - [ParametersForModeTeff]
    - `bounds_teff` are the bounds for temperature (HARD BOUNDS)
    - `guess_range_teff` is the range of temperature for the initial guess deviated from the input temperature
  - [Bounds]
    - Bounds for vmac, rotation, abundance and doppler shift (deviated from RV)
  - [GuessRanges]
    - Guess ranges for vmac, rotation, abundance and doppler shift (deviated from RV)
  - [SlurmClusterParameters]
    - See more details in [## Multiprocessing usage](##multiprocessing-usage)

---

## Plotting Tools
The Jupyter notebook `plot_output.ipynb` (in `plotting_tools/`) helps you visualize fitted lines and results:
1. **Make a copy**: `cp plot_output.ipynb plot_output_test.ipynb`
2. **Open** `plot_output_test.ipynb` and run the first cell to import libraries.
3. **Change** `output_folder_location` to your results folder.
4. **Run** the subsequent cells to generate:
   - Per-line or per-star plots
   - Comparisons of observed vs. best-fit synthetic spectra
   - Additional diagnostic plots (e.g., residuals)
5. **Optional**: Synthetic spectrum generation is showcased in the last few cells.

---

## Compiler Notes (ifort / ifx)

- **ifort** (not supported anymore) or **ifx** from Intel can be installed via [oneAPI Fortran compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/fortran-compiler.html).
- On Mac, you may need to source `setvars.sh` to make `ifort` or `ifx` available.
- gfortran is also supported. The `mcmodel=medium` Makefile edit (line 11) may be required on some Linux systems for large arrays.

---

## Parallelisation and Clusters
TSFitPy uses **Dask** for parallel processing:

- **Local Mode**: simply set `number_of_cpus` in your config to however many cores you want. The code automatically uses your CPU cores.
- Locally, you can simply run `http://localhost:8787/` in your browser (port might be different, but probably not)
- **Slurm / HPC**: advanced usage is supported by setting `cluster_type = slurm` in `[SlurmClusterParameters]` of the config. Then specify:
  - `number_of_nodes`, `memory_per_cpu_gb`, `partition`, `time_limit_hours`, etc.
  - `script_commands` are the commands to run before the script. Each command is separated by a semicolon. Example below purges the modules and loads the ones needed
    - `module purge;module load basic-path;module load intel;module load anaconda3-py3.10`
  - The script will act as a “manager” job that spawns multiple Slurm workers. Each worker runs in parallel.
  - Use `cluster_name` to help forward the Dask dashboard port, so you can monitor progress in your browser.
- If ran on a cluster, you can use SSH command to direct this dashboard to your own PC
  - For that you write the name of your cluster (whichever server name you connect to) in the cluster_name in the [ExtraParameters]
  - This will print something similar to: `ssh -N -L {port}:{host}:{port} {cluster_name}`, where cluster_name is taken from your config
  - It should automatically figure out your host and port where the dask dashboard is ran
  - By running this command in your terminal, it will redirect the dashboard to your browser with port port
  - So you can once again check the dashboard in your browser by running `http://localhost:{port}/`, replacing `{port}` with the port above

---

## Troubleshooting & Flags
TSFitPy assigns two 8-bit flags in the output:

- **`flag_error`**: indicates critical issues. If any bit is set, treat the result with skepticism.
  - **Bit 1**: The fit failed to converge.
  - **Bit 2**: Insufficient observed data points in the line mask (<=2 points in the spectra).
  - **Bit 3**: Observed flux is entirely above 1 or below 0.
  - **Bit 4**: Extreme EW mismatch between observed line vs model (within factor of 1.5).
  - **Bit 5**: Very low iteration count during the fit (<= 3).
  - **Bit 6**: When calculating spectra with offset abundance or no abundance (options `compute_blend_spectra` and `sensitivity_abundance_offset`), the chi squared should be higher than the chi squared of the fit with abundance. If it is not, then it is likely that the minimum is not found and triggers this flag.
  - **Bit 7**:
  - **Bit 8**:

- **`flag_warning`**: potential but not certain problems. It doesn't mean that the fit is bad, but it is worth checking.
  - **Bit 1**: Fitted parameter is at a boundary limit.
  - **Bit 2**: Observed flux is partially above 1.1 or below 0.
  - **Bit 3**:
  - **Bit 4**: Moderate EW mismatch between observed line vs model (within factor of 1.25).
  - **Bit 5**: Possibly too few iterations (<= 5).
  - **Bit 6**:
  - **Bit 7**:
  - **Bit 8**:

Check flags, visually inspect results, and confirm your config.

---

## Conclusion
Thank you for using **TSFitPy**! We hope these instructions help you install, fit, and interpret your spectroscopic data effectively. If you have additional questions, feel free to open an issue on the [TSFitPy GitHub repository](https://github.com/TSFitPy-developers/TSFitPy/).

---

## Extra notes

- Here is the (not very up-to-date) Trello board for the project: https://trello.com/b/2xe7T6qH/tsfitpy-todo

---

## FAQ

- **Support for non MARCS models?**
  - Currently, only MARCS models are supported. You can probably use other models by using them as <3D> models, with the same format as them. But NLTE is only calculated for 16k standard MARCS models and <3D> STAGGER ones
- **Using non-standard composition MARCS models?**
  - See [this](https://github.com/TSFitPy-developers/TSFitPy/issues/76)
- **Reference for the vmic relation?**
  - See [this](https://github.com/TSFitPy-developers/TSFitPy/issues/80), i.e. there is no published work
- **Fitting isotopes?**
  - See [this](https://github.com/TSFitPy-developers/TSFitPy/issues/73)
- **How is fitting done exactly?**
  - Typically for line-by-line (`lbl`) fit, each line is fit separately on each CPU in the following manner:
    1. Generate spectra for the specific abundance using TS (assuming constant stellar parameters). Steps: interpolate model atmosphere (+NLTE departure coefficients if relevant), run babsma, run bsyn. This step is done at every Nelder-Mean step.
    2. Fit the generated spectrum, by changing vmac, rotation, RV using another minimisation algorithm (`L-BFGS-B`). This method works better here because we can take a very small step for gradient calculations. 
    3. Therefore, for the specific abundance generated in step 1, we find best-fit vmac/rotation/RV. This breaks degeneracy of fitting both broadening and abundance (way more accurate).
    4. Repeat steps 1-3 until best-fit abundance is found. This is done by changing the abundance in the generated spectrum and repeating the fit.
    5. Once the best-fit abundance is found, we can calculate the equivalent width of the line and do other analysis (generate spectra of just the blends, by changing abundance a bit etc.).

---

## Some debugging tips:

- How to debug the code:
  - Run with 1 CPU only
  - Set `debug=2`
    - This will print out the Fortran output and extra Python output
  - Look at the bottom of the output: usually you will see what goes wrong
    - You can also search for `forr`, because Fortran errors usually start with `forrtl`


- TurboSpectrum has a limit of 100 linelists. So if you have too many linelists, it will crash.
  - In that case combine some of the linelists into one


- If you get a Fortran error `forrtl: severe (24): end-of-file during read, unit -5, file Internal List-Directed Read` in the `bsyn_lu            000000000041DB92  getlele_                   38  getlele.f` just after it trying to `  starting scan of linelist` with some molecular name, then the issue is probably the following one:
  - Your abundance of the element is too low (e.g. I had that issue with [X/Fe] = -30) and it skips that element in the molecular line identification. In that case remove the molecular linelist containing that element, or increase your element abundance (e.g. to [X/Fe] = -3)