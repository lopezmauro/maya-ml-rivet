
/*This module is mean to be used to get the main training data for train the model to be used on ml_rivets.mll node
This code is to be used on maya with numpy library

MIT License

Copyright (c) 2020 Mauro Lopez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "CNTKLibrary.h"
#include <string.h>
#include <fstream>
#include <maya/MPxNode.h> 
#include <maya/MGlobal.h> 
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnEnumAttribute.h>

#include <maya/MFnTypedAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MString.h> 
#include <maya/MTypeId.h> 
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MMatrix.h>
#include <maya/MVector.h>
#include <valarray> // for value array operations 

#define INMEAN "in_mean.csv"
#define INSTD "in_std.csv"
#define OUTMEAN "out_mean.csv"
#define OUTSTD "out_std.csv"


class mlRivet : public MPxNode
{
public:
	mlRivet();
	~mlRivet() override;

	MStatus	compute(const MPlug& plug, MDataBlock& data) override;
	MStatus jumpToElement(MArrayDataHandle& hArray, unsigned int index);

	static  void*		creator();
	static  MStatus		initialize();

	CNTK::FunctionPtr getModel(const std::wstring newFilePath);
	std::map<std::string, std::valarray<float>> getDataMap(const std::wstring inDataPath);

public:
	
	static	MObject		deviceType;			// GPU or CPU.
	static	MObject		modelFilePath;		// The trained model path.
	static	MObject		inDataFilePath;
	static	MObject		inputs;

	static	MObject		matrix;
	static	MObject		outputs;

	static  bool		_debug;
	static	MTypeId		id;

private:
	std::wstring modelPath;
	std::wstring dataPath;
	CNTK::FunctionPtr modelPtr;
	std::map<std::string, std::valarray<float>> dataMap;
};

