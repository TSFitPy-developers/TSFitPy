# Turbospectrum Spectral Fitting with Python (TSFitPy)
<!--# AKA PODRACES (Pipeline for the Objective Determination of Realistically Accurate Characteristics and Elements of Stars)-->

TSFitPy is a pipeline designed to determine stellar abundances and atmospheric parameters through the use of Nelder-Mead (simplex algorithm) minimization. It calculates model spectra "on the fly" while fitting instead of using a more sophisticated method that relies on training neural networks such as the method used by the full SAPP used for much larger datasets. Using this method allows the pipeline to gain flexibility in the stellar physics used at the expense of computation time, which makes it useful for small datasets of ~100 stars and fewer.

To use TSFitPy, you will need a working Turbospectrum (TS) installation of the latest version, which has the capability to compute NLTE line profiles as well as calculate specified spectral windows instead of a full spectrum for a given range. TSFitPy has not been tested on older versions of TS.

The code is written in Python 3 (3.7.1, but should work with any version of python 3). It also makes use of fortran programs, which will need to be compiled on the 
user's machine. The Python packages needed are as follows:
- Numpy
- Scipy
- Time
- OS
- Subprocess
- Glob
- Re
- Math
- Operator

They should all be installable via "pip install"

All of the fortran codes are compilable either with a gnu or ifort compiler. In the scripts folder, there is a Python script titled "compile_fortran_codes.py". Running this code should compile all of the necessary codes needed for the main TSFitPy pipeline. It makes use of the OS Python package.

TSFitPy uses a system of relative directories to run Turbospectrum and know where to find various inputs that control the fitting procedure. These directories need to be preserved in order to run correctly. The main directories in use are input_files, output_files, scripts, turbospectrum, and the various folders within. A working Turbospectrum installation needs to be installed in the "turbospectrum" directory so that the executive files for Turbospectrum are in a directory "turbospectrum/exec/" (or in "turbospectrum/exec-gf/" if using the gnu compiler).

Other directories that require user input are the "model_atmospheres" and "linelists" directories in input_files. The model_atmospheres has two directories, one called "1D" for MARCS model atmospheres (include reference), and one called "3D" for average 3D model atmospheres (include info about STAGGER grid). A list of all model atmospheres in the folder called "model_atmosphere_list.txt" also needs to be in each folder. An example is provided to show the formatting.

The linelist/s used by Turbospectrum should be put in the folder "linelists/linelist_for_fitting/". Some example linelists are included that were made from the VALD3. In addition, various atomic and molecular linelists used by the Gaia-Eso survey can be found here https://keeper.mpdl.mpg.de/d/4b475c7472e54b01a845/. The atomic lists have been cross-matched with the model atoms used for NLTE to identify transition levels for the lines and can therefore be used for NLTE computations with Turbospectrum. We have supplemented these lists with lines pulled from the VALD3 for lines falling outside of the Gaia-ESO range (4200 - 9200 AA). These files cannot end in ".txt".

Finally, if you wish to fit using NLTE, you'll need the relevant NLTE data. This includes binary files of departure coefficients, auxiliary text files that tell the model interpolators how to read these files, and model atom files. These files are too large to keep on Github (especially the binary files), so those available to the public can be found at https://keeper.mpdl.mpg.de/d/9b6c265057aa4bceb939/. The information then needs to be stored in the relevant directories under "input_files/nlte_data".

Once these folders are set, you can begin fitting with TSFitPy. Set up the parameters of your fit using the "tsfitpy_input_configuration.txt" file, place normalized observed spectra for fitting in the folder "observed_spectra" in input_files, and update the "fitlist" file to list the spectra to be fit and their parameters. The folder "examples" provides some examples of various fitting methods as well as sample input files, output files, and terminal outputs. These examples were done using the VALD3 example linelists.
