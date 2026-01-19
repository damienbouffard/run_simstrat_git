# Simstrat on Renkulab

## RUN Test Case

1. Open the terminal
2. Enter the test directory `cd tests`
3. Run the test .sh script `./run_testcase.sh`


## RUN Your simulation

1. upload the forcing files and the configuration file a Renkulab session See forin instance Models Input on [Alplakes](https://www.alplakes.eawag.ch/downloads )
2. Open the terminal
3. Enter the test directory `cd simulation`
4. Run the model using .sh script `./run_testcase.sh`


## MODIFY forcing

1. You can modify the forcing using the jupyter notebook provided `notebooks`.   
a.  `Modify_Simstrat_Forcing.ipynb` will help you modify the meteorological forcing
b.  `Modify_Simstrat_Inflow.ipynb` will help you modify the inflow forcing
2. modify the par file with the new forcing conditions
3. provide a new path and and new name for the model output
4. copy `./run_testcase.sh` to the new path and edit the name of the par file in the .sh script
5. Run the model

