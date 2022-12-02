# Fisher Criterion Beamformer - Statistical spatial filter for Event Related Potentials 

This archive provides a Toolbox (Matllab and Python versions) implementing a Statistical Spatial Filter called Fisher Criterion Beamformer, specifically designed to improve the discrimination of Event Related Potentials (ERPs), proposed in [1] (section 3.2.2):

> [1] Gabriel Pires, Urbano Nunes and  Miguel Castelo-Branco (2011), "Statistical Spatial Filtering for 
   a P300-based BCI: Tests in able-bodied, and Patients with Cerebral Palsy and Amyotrophic Lateral 
  Sclerosis", Journal of Neuroscience Methods, Elsevier, 2011, 195(2), 
  Feb. 2011: doi:10.1016/j.jneumeth.2010.11.016
  https://www.sciencedirect.com/science/article/pii/S0165027010006503?via%3Dihub

## Example
A self-contained example is provided using 2 different datasets that can be used for testing FCB toolbox:
> dataset 1 : P300 oddball elicited in a communication speller (LSC speller)
> dataset 2 : Error-related potentials (ErrP) elicited to detect and correct errors in LSC speller

For more information about LSC speller and datasets please check: 
> [2] https://ieee-dataport.org/open-access/error-related-potentials-primary-and-secondary-errp-and-p300-event-related-potentials-%E2%80%93
> [3] https://ieeexplore.ieee.org/abstract/document/8048036

### For the purpose of this example, it is useful to know that the datasets use:   
> 12 EEG channels: order [1-Fz 2-Cz 3-C3 4-C4 5-CPz 6-Pz 7-P3 8-P4 9-PO7 10-PO8 11-POz 12-Oz]
> sampling frequency: fs = 256 Hz
> dataset 1 is filtered [0.5-30] Hz
> dataset 2 is filtered [1-10] Hz
> epochs are all 1 second long

## Code
> the FCB toolbox is in folder 'FCB_toolbox' 
> Some analysis files are in folder 'rsquare'
> The main file is main.m (Matlab version) or main.py (Python version) 

This example is self-explanatory from code and comments - just run and select the dataset you want to test

To show the discrimination effect of FCB,  statistical r-square is applied on data before and after FCB filtering 

## Paper(s) and dataset(s)
If you use the FCB toolbox please refer to [1]
> [1] Gabriel Pires, Urbano Nunes and  Miguel Castelo-Branco (2011), "Statistical Spatial Filtering for 
   a P300-based BCI: Tests in able-bodied, and Patients with Cerebral Palsy and Amyotrophic Lateral 
  Sclerosis", Journal of Neuroscience Methods, Elsevier, 2011, 195(2), 
  Feb. 2011: doi:10.1016/j.jneumeth.2010.11.016
  https://www.sciencedirect.com/science/article/pii/S0165027010006503?via%3Dihub

If you use these datasets please refer to [2] and [3]
> [2] https://ieee-dataport.org/open-access/error-related-potentials-primary-and-secondary-errp-and-p300-event-related-potentials-%E2%80%93
> [3] https://ieeexplore.ieee.org/abstract/document/8048036
