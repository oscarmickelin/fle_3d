# fle_3d

Code accompanying the paper:

If you use the code, please cite the above paper.

## Installation
```bash
# install general dependencies
pip install numpy scipy finufft joblib

# install fast spherical harmonics transform
pip install torch-harmonics

#########
# alternative fast spherical harmonics transform (optional)
pip install juliacall
python3 install_julia_transforms.py
#########

# run test code (optional)
python3 test_fle_3d.py
```
