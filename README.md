# Maya-ml-rivet

This is a proof of concept of a Maya rivet node done with machine learning. Is not meant to be an optimized code.
THIS IS A BETA RELEASE USE AT YOUR OWN RISK

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

Using this method I added 40 rivet to the face of mary rig without any any performance loss
![](mery_rivets.gif)


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
### Getting the data

In order to get the data you need to specify the driver controls (control that deform the mesh), the driven transforms (controls that will be riveted) and the mesh name. This is an example of getting the data for a bunch of transforms that will be rivetted to the face of [mery](https://www.meryproject.com/) (special thanks to Antonio Francisco Méndez Lora for provide the rig)
```python
import sys
#add the path where the code is to the system path
sys.path.append(r'D:/dev/MayaNodes/ml_rivet')
from pyutils import getRivetsSceneData
mesh = 'Mery_geo_cn_body'
# the control list will filter itself for the relevant controls
driverList = [u'Mery_ac_rg_stickyLips_control', u'Mery_ac_lf_stickyLips_control', u'Mery_ac_lf_tinyCorner_control', u'Mery_ac_rg_tinyCorner_control', u'Mery_ac_loLip_01_control', u'Mery_ac_loLip_02_control', u'Mery_ac_loLip_03_control', u'Mery_ac_loLip_04_control', u'Mery_ac_loLip_05_control', u'Mery_ac_upLip_05_control', u'Mery_ac_upLip_04_control', u'Mery_ac_upLip_03_control', u'Mery_ac_upLip_02_control', u'Mery_ac_cn_inout_mouth', u'Mery_ac_upLip_01_control', u'Mery_ac_rg_cheekbone', u'Mery_ac_lf_cheekbone', u'Mery_ac_cn_mouth_move', u'Mery_ac_dw_lf_lip_inout', u'Mery_ac_up_lf_lip_inout', u'Mery_ac_dw_rg_lip_inout', u'Mery_ac_lf_moflete', u'Mery_ac_rg_moflete', u'Mery_ac_up_rg_lip_inout', u'Mery_ac_cn_jaw_control', u'Mery_ac_jaw_front', u'Mery_ac_lf_corner_control', u'Mery_ac_lf_nose', u'Mery_ac_rg_corner_control', u'Mery_ac_rg_nose']
#I created a bunch of spheres in a group called drivens
drivenList =[cmds.listRelatives(a, p=1)[0] for a in cmds.listRelatives('drivens', ad=1, type='mesh')]
folderData = r'D:\projects\ml_rivet\data'
filePrefix = 'mery_'
samples = 300
getRivetsSceneData.getData(mesh, driverList, drivenList, folderData, filePrefix, samples)
```
this will generate 3 files: 
('D:\\projects\\ml_rivet\\data\\mery_inputs.csv', 'D:\\projects\\ml_rivet\\data\\mery_outputs.csv', 'D:\\projects\\ml_rivet\\data\\mery_transforms.json')

## training the model
On a command line excecute trainModel.py file.
If you have python 2 and 3 you need to specyfy the python version (py -3), you couls use -h flag to het the command help
```
D:\dev\MayaNodes\ml_rivet\pyutils> py -3 trainModel.py -h
```
On my example I used
```
D:\dev\MayaNodes\ml_rivet\pyutils> py -3 trainModel.py -o "D:/projects/ml_rivet/model" -d "D:/projects/ml_rivet/data" -p "mery_"
```
## Loading the model back to maya
In order to load the model back you need:
* to load the plugin
* create a ml_rivet node
* set the modelFile and dataFile attributes
* connect the drivers and the drivens (the transforms.json saved with the data has that information)

```python
import sys
sys.path.append(r'D:\dev\MayaNodes\ml_rivet')
from pyutils import getRivetsSceneData
modelFile = r"D:/projects/ml_rivet/model/mery_model.onnx"
dataFile = r"D:/projects/ml_rivet/model/mery_modelData.zip"
dataJson = r"D:/projects/ml_rivet/data/mery_transforms.json"
cmds.file(new=1,f=1)
cmds.unloadPlugin(r'ml_rivet.mll')
cmds.file('D:/projects/ml_rivet/Mery_v3.5_for_2015_rivetsStart.mb', o=1, f=1)
cmds.loadPlugin(r'D:\dev\MayaNodes\ml_rivet\release\ml_rivet.mll')
node = cmds.createNode('ml_rivet')
cmds.setAttr('{}.modelFile'.format(node),modelFile , type="string")
cmds.setAttr('{}.dataFile'.format(node), dataFile , type="string")

jsonData = getRivetsSceneData.readJsonFile(dataJson)
for x, ctrl in enumerate(jsonData.get('drivers')):
    cmds.connectAttr('{}.xformMatrix'.format(ctrl), '{}.inputs[{}]'.format(node,x))
for i, drv in enumerate(jsonData.get('drivens')):
    dec = cmds.createNode('decomposeMatrix')
    cmds.connectAttr('{}.outputs[{}]'.format(node, i), '{}.inputMatrix'.format(dec))
    cmds.connectAttr('{}.outputTranslate'.format(dec), '{}.translate'.format(drv))
```

### how evaluate the model accuracy
Using the flag -v -verbose on the train model command line you can see the accuracy and loss details every 50 samples.
- The lower the loss, the better a model
- val_loss is the loss but on unseen samples, if this value dont lower at the same rate than the loss it means that the model over-fitted

### how to get better acurracy
increasing the **sampling** on getRivetsSceneData.getData (getting more data to the model to learn) or increasing the **epochs** on trainModel (forcing to train more times over the same data)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

