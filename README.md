# GPJet

Learning of E-Jet Printing Dynamics with Physics-Informed Gaussian Processes

<hr>

## Part I: Computer Vision Module

Description:<br>
A real time computer vision framework using OpenCV for 
real-time feature extraction relevant to MEW process dynamics.

- code: [LINK](code/computer-vision-module/code)
    - sequential
    - concurrent 
    - parallel
- data: [LINK](https://drive.google.com/drive/folders/1T5KFV0BZIe3SvaFVtMDVANbrJmCn5hfI?usp=sharing)
    - pre-processed videos for JM framework 
- results: [LINK](https://drive.google.com/drive/folders/1Dl1Jc7Z1xCP-lMfqmkhcrHiPVb4ccRSV?usp=sharing)
    - processed videos after JM framework
    - jet feature extraction
  
## Part II: Machine Learning Module

Description:<br>
Surrogate modeling using jet metrology data.

- code: [LINK](code/machine-learning-module)
  - GP Regression (GPR) -> Surrogate Models
  - Multi-Fidelity (MFD) -> Surrogate Models & Physics Models
  - Active Learning 1 (AL1) -> Surrogate Models
  - Active Learning 2 (AL2)  -> Multi-Fidelity Models
- dataset:
  - GPR & AL1: [LINK](https://drive.google.com/drive/folders/1tFHWmyanFhN-GnEjM4IRvT9cyAWBOr6z?usp=sharing)
  - MFD & AL2: 
    - High fidelity dataset from Computer Vision Module:
      - [LINK](https://drive.google.com/drive/folders/1tFHWmyanFhN-GnEjM4IRvT9cyAWBOr6z?usp=sharing)
    - Low fidelity dataset from Jet Diameter Model:
      - [LINK](https://drive.google.com/drive/folders/1ECz8zNblBrMCLsar_Xa_uMdPCE_7fHOt?usp=sharing)
- results: [LINK](gpr/results)

## Part III: Physics Module

Description:<br>

- Jet Diameter Model: [LINK](code/physics-module/)
  - Results: [LINK](https://drive.google.com/drive/folders/1ECz8zNblBrMCLsar_Xa_uMdPCE_7fHOt?usp=sharing)
- Mechanical Fluid Sewing Machine Patterns: [LINK](code/physics-module)
  - Results: [LINK](https://drive.google.com/drive/folders/12BEIVG5AojBvJAgc5FIWn1XFlJVCJ3dS?usp=sharing)


