# Introducing a computational model of aesthetic value
Repository for code and results associated with "Introducing a computational model of aesthetic value" (Brielmann &amp; Dayan, in prep.)
The scripts stored in this repository can be used to recreate the results and figures presented in this paper. 

The functions defined in "python_packages/aestheticsModel/simExperiment.py" can also be used as a tool for flexible application of the proposed model of aesthetic value to fit other datasets or simulate model predictions for a variety of other experimental paradigms.

## fitting existing data 
The folder "fit_existing_data" contains the scripts to fit meta-analytic findings from Montoya et al. (2017) and Tioni & Leder (2009). Either script can be run from its location in the folder structure given. Both require that the folder "pyhton packages" and its contents are located in the same relative location as in this repository.

## re-creating paper figures
The folder "create_figures" contains the scripts for creating all figures displayed in the main article. Either script can be run from its location in the folder structure given. Both require that the folder "results" and its contents are located in the same relative location as in this repository.

## results
The "results" folder contains the results of the model fitting reported in the main paper in .csv format. 
Model-fitting results can be re-created by running the scripts contained in the "fit_existing_data" folder. Results can be saved and are by default saved to the same folder as the scripts that contain them. Please note that the saved .csv files do not contain the header; for approproiate naming of the .csv file columns, please refer to the .csv files already provided in the "results" folder.

---

 
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
