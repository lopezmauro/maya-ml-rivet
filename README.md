# maya-ml-rivet

Proof of concept of a maya rivet node done with machine learning. 

### Motivation
Good portion of the performance of a  on a rigged character is lost on the rivets (ussually added for the facial controls). 
Vecause you want to the rivets follow all the deformation stack, is ussully added on top of the mesh that recive all the deformation. This instroduce three main bottle necks:
* All the deformation stack need to be evaluated to compute the rivet final position
* Adding the transform dependency to the mesh, makes most of the deformers be out of the GPU computation
* force to negate the riveted control deformation by adding a inverse transformation on the control's parent

### Posposal
Train an Linar regresion model that learn the mesh deformation, then a custom node use that module to predict the control position when the mesh defomrs. Allowing to:
* compute outside the deformation stack
* break the compute dependency to the mesh allowing all deformers compute on the GPU
* Skip the riveted controls on the training so it own deformation dont needs to be reversed

# Using ml_rivet.mll
## Before start

### Pre Compiled version

A precompiled Windows version for maya2019 is provided on the release folder. The model train is done un pyTorch, but the easiest alternative to load it in C++ on Windows is Microsoft's CNTK. using [ONNX](http://onnx.ai/) as exchange format. So to use the precompiled plugin, you will need to donwload [Cognitive Toolkit](https://cntk.ai/dlwg-2.7.html), un zip it and copy all dll fils from cntk folder to a place where maya find is. If you are unsure just place them along with the ml_rivets.mll

### Compile you own version

Beside the maya libraries dependencies you need:

* [CNTK](https://docs.microsoft.com/es-es/cognitive-toolkit/) 
* [zlib](https://zlib.net/)
* [libzip](https://libzip.org/)

You can install CNTK by right click the References in your project and start NuGet, and seach Search for CNTK, you may want to use the GPU version if you have a compatible graphic card

### Train Model Prerequisites

All the data collection and trainning is done in python using pyTorch, but becasue pyTorch need python3 and maya still uses python 2.7 the procces nees to be done in two steps, data collection inside maya and model trining outside.

For data collection in maya maya you only need numpy:
https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722

For model train you will need:
* [python 3.7+](https://www.python.org/downloads/windows/) - python 3.7 or newer
* [pytorch](https://pytorch.org/get-started/locally/)  - pythorch cuda version recomened (if you have compatible GPU)
* [tqdm](https://pypi.org/project/tqdm/) - nice procress bar


## Running instrucions

Coming soon...

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

