# Maya-ml-rivet

Proof of concept of a Maya rivet node done with machine learning.

### Motivation

A good portion of the performance of an  on a rigged character is lost on the rivets (usually added to the facial controls). Because you want to the rivets follow all the deformation stack, is usually added on top of the mesh that receive all the deformation. This introduces three main bottlenecks:
* All the deformation stack need to be evaluated to compute the rivet final position
* Adding the transform dependency to the mesh, makes most of the reformers be out of the GPU computation
* force to negate the riveted control deformation by adding an inverse transform on the control's parent

### Posposal
Train a Linear Regression Model that learn the mesh deformation, then a custom node use that Model to predict the control position when the mesh deforms. Allowing to:
* compute outside the deformation stack
* break the compute dependency to the mesh allowing all deformers compute on the GPU
* Skip the riveted controls on the training so it own deformation doesn't need to be reversed

# Using ml_rivet.mll
## Before start

### Pre Compiled version

A precompiled Windows version for maya2019 is provided in the release folder. The model train is done on pyTorch, but the easiest alternative to load it in C++ on Windows is Microsoft's CNTK. using [ONNX](http://onnx.ai/) as an exchange format. So to use the precompiled plugin, you will need to download [Cognitive Toolkit](https://cntk.ai/dlwg-2.7.html), unzip it and copy all .dll files from cntk folder to a place where Maya find them. If you are unsure just placing them along with the ml_rivets.mll

### Compile you own version

Beside the Maya libraries dependencies, you need:

* [CNTK](https://docs.microsoft.com/es-es/cognitive-toolkit/) - Microsoft Cognitive Network Toolkit
* [zlib](https://zlib.net/) - lossless data-compression library
* [libzip](https://libzip.org/) - Library for reading, creating, and modifying zip archives 

You can install CNTK by right click the References in your project and start NuGet, and search Search for CNTK, you may want to use the GPU version if you have a compatible graphic card

### Train Model Prerequisites

All the data collection and training are done in python using pyTorch, but because pyTorch needs python3, and Maya still uses python 2.7 the process needs to be done in two steps, data collection inside Maya and model training outside.

For data collection in Maya, you only need numpy:
https://forums.autodesk.com/t5/maya-programming/guide-how-to-install-numpy-scipy-in-maya-windows-64-bit/td-p/5796722

For model train you will need:
* [python 3.7+](https://www.python.org/downloads/windows/) - python 3.7 or newer
* [pytorch](https://pytorch.org/get-started/locally/)  - pythorch cuda version recomened (if you have compatible GPU)
* [tqdm](https://pypi.org/project/tqdm/) - nice progress bar


## Running instrucions

--- Coming soon ...

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

