#This file installs juliacall from https://juliapy.github.io/PythonCall.jl/stable/juliacall/
#First run pip install juliacall,
#then run the below to add the packages used for
#asymptotically fast spherical harmonics transform.

import juliapkg

juliapkg.add("FastTransforms", "057dd010-8810-581a-b7be-e3fc3b93f78c")
juliapkg.add("FastSphericalHarmonics", "d335c211-7587-445e-a753-f0b1fb7e445f")
juliapkg.resolve()
