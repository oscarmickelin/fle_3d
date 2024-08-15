# fle_3d

Code accompanying the paper: https://arxiv.org/abs/2406.05922

If you find the code useful, please cite the above paper.

## Installation
```bash
# install general dependencies
pip install numpy scipy finufft

# install fast spherical harmonics transform
pip install torch-harmonics

#########
# install alternative fast spherical harmonics transform (optional)

pip install juliacall
python3 install_julia_transforms.py
#########


#########
# install dependencies to create dense matrix operators,
# to check accuracy of the fast methods
# (optional, but required to run the
# first and third tests in test_fle_3d.py)

pip install joblib mrcfile tqdm

# to create the dense matrix operators for N > 32,
# need to additionally install pyshtools
# Note: highly optional. The only reason to do this
# is to be able to run the accuracy tests for large N.
# Apart from this, pyshtools is not used in the code
# and there is no need to install it

pip install pyshtools
#########

# run test code (optional)
python3 test_fle_3d.py
```

## Usage

Given a volume x represented by a 3D array of size NxNxN that you want to expand into the ball harmonic basis, first create a basis object by calling
```python
from fle_3d import FLEBasis3D
N = 128         #replace this by the side-length of your volume array
bandlimit = N   #maximum number of basis functions to use
eps = 1e-7      #desired accuracy
fle = FLEBasis3D(N, bandlimit, eps)
```
Here, eps is the accuracy desired in applying the basis expansion, corresponding to the epsilon in Theorem 3.1 in the paper. "Bandlimit" is a parameter that determines how many basis functions to use and corresponds to the variable lambda in equation (40) in the paper, scaled so that N is the maximum suggested.

All arguments to FLEBasis3D:

- N:    size of volume to be expanded

- bandlimit:    bandlimit parameter (scaled so that N is max suggested)

- eps:     requested relative precision

- maxitr:      maximum number of iterations for the expand method (if not specified, pre-tuned values are used)

- maxfun:      maximum number of basis functions to use (if not specified, which is the default, the number implied by the choice of bandlimit is used)

- max_l:           use only indices l < max_l, if not None (default).

- mode:       choose either "real" or "complex" (default) output, using either real-valued or complex-valued basis functions

- force_real:   If true, get a speedup by a factor 2 by enforcing that the source (for evaluate_t) or target (for evaluate) is real. To reproduce the tables and figures of the paper, set this to True.

- sph_harm_solver: solver to use for spherical harmonics expansions.
                Choose either "nvidia_torch" (default) or "FastTransforms.jl".
                
- reduce_memory: If True, reduces the number of radial points in defining
                NUFFT grids, and does an alternative interpolation to
                compensate. To reproduce the tables and figures of the
                paper, set this to False. 
    
To go from the volume to the basis coefficients, you would then call either

```python
coeff = fle.evaluate_t(x)
```

which applies the operator $\tilde{B}^*$ in Theorem 3.1 of the paper, or 

```python
coeff = fle.expand(x)
```
which solves a least squares problem instead of just applying equation $\tilde{B}^*$ once. The latter can be more accurate, but takes a bit longer since it applies evaluate_t ```maxitr``` times using Richardson iteration.

Once you have coefficients ```coeff``` in the basis, you can evaluate the corresponding function with expansion coefficients ```coeff``` on the NxNxN grid by running

```python
volume = fle.evaluate(coeff)
```

which corresponds to applying the operator $\tilde{B}$ in Theorem 3.1 in the paper.
