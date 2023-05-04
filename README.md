![Banner](img/Banner.png)

Learning of E-Jet Printing Dynamics with Physics-Informed Gaussian Processes

This Github repository contains all the data and code needed to replicate the results reported 
in this paper:   

[Oikonomou et al. (2023), "Physics-Informed Bayesian learning of electrohydrodynamic polymer jet printing dynamics", Communications Engineering, Nature](https://www.nature.com/articles/s44172-023-00069-0)

## GPJet Code
Learn how to set up the environment, run and use the code: [HERE](code/)  

To run the GPJet framework download or clone the github repo and install packages as 
referred at `gpjet.yml` 

Otherwise, open anaconda prompt and run:

`conda env create --file gpjet.yml`

## Part I: Computer Vision Module

Description:<br>
A real time computer vision framework using OpenCV for 
real-time feature extraction relevant to MEW process dynamics.

- code: [LINK](code/computer-vision-module/code)
    - sequential
    - concurrent 
    - parallel
- data: [LINK](https://drive.google.com/drive/folders/1kFALwARpGhZ1VKI2ImHqpedA_1iNfm9Y?usp=share_link)
    - initially processed videos for JM framework (kommena ta aspra panw katw) 
- results: [LINK](https://drive.google.com/drive/folders/1tQYqzo4206dvHiCcaMSUHDQfxvBOK3kd?usp=share_link)
    - processed videos after JM framework
    - these videos were used for jet feature extraction
  
## Part II: Machine Learning Module

Description:<br>
Surrogate modeling using jet metrology data.

- code: [LINK](code/machine-learning-module)
  - GP Regression (GPR) -> Surrogate Models
  - Multi-Fidelity (MFD) -> Surrogate Models & Physics Models
  - Active Learning 1 (AL1) -> Surrogate Models
  - Active Learning 2 (AL2)  -> Multi-Fidelity Models
- dataset:
  - GPR & AL1: [LINK](https://drive.google.com/drive/folders/1Xgb0l8LmB0Q7JQCHCJbmZAv1CEXCWetD?usp=share_link)
  - MFD & AL2: 
    - High fidelity (extracted jet features) dataset from Computer Vision Module:
      - [LINK](https://drive.google.com/drive/folders/1Xgb0l8LmB0Q7JQCHCJbmZAv1CEXCWetD?usp=share_link)
    - Low fidelity dataset from Jet Diameter Model:
      - [LINK](https://drive.google.com/drive/folders/1i4h7oKqxGQJ2gbZmZluoAWMdpjW_O9XB?usp=sharing)
- results: [LINK](...)

## Part III: Physics Module

Description:<br>

- Jet Diameter Model: [LINK](code/physics-module/)
  - Results: [LINK](https://drive.google.com/drive/folders/1ECz8zNblBrMCLsar_Xa_uMdPCE_7fHOt?usp=sharing)
- Mechanical Fluid Sewing Machine Patterns: [LINK](code/physics-module)
  - Results: [LINK](https://drive.google.com/drive/folders/12BEIVG5AojBvJAgc5FIWn1XFlJVCJ3dS?usp=sharing)
