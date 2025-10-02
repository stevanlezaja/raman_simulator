# 02.10

## Added
* Controller Base class
* Control Loop class
* Gain custom type
* Raman Inputs and Spectrum classes
* Generic Unit addition
* Spectrum addition and subtraction
* Method for adding frequency-value pair to Spectrum
* Experiment script for spectrum amplification
* PID Controller class
* Script for testing controllers
* Custom logger class
* Controller tests
* Raman Inputs tests
* Signal properties to access power and wavelength
* Raman System class
* Methods for getting and setting values at frequencies in Spectrum
* Bernoulli controller class
* Script for testing spectrum control
* Addition operation for Raman Inputs
* Clamping Raman Inputs to target range
* Control Loop tests
* properties for checking if class is initialized properly (valid)

## Changed
* Strong typing and linting in custom types
* Strong typing and linting in Fiber
* Strong typing and linting in Raman Amplifier and Pump
* Experiment class updates solution when a method is called instead of only on init
* Strong typing and linting in Raman Inputs
* Raman Inputs init optionally populates values
* Using logging instead of printing - easier filtering and debugging
* scripts now just provide a main function that is to be called from `main.py`
* Strong typing and linting in Controller Base
* Strong typing and linting in Raman Inputs and Spectrum
* Spectrum iteration iterates over sorted frequencies
* Units check for negative value and clip accordingly

## Fixed
* Power Gain custom type default unit `''` -> `' '`

# 18.09