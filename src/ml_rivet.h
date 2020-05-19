
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


class ModelCache
{
	std::wstring modelPath;
	std::wstring dataPath;
	CNTK::FunctionPtr modelPtr;
	std::map<std::string, std::valarray<float>> dataMap;

public:
	CNTK::FunctionPtr getModel(const std::wstring newFilePath);
	std::map<std::string, std::valarray<float>> getDataMap(const std::wstring inDataPath);


	ModelCache();
	~ModelCache();
};


class mlRivet : public MPxNode
{
public:
	mlRivet();
	~mlRivet() override;

	MStatus	compute(const MPlug& plug, MDataBlock& data) override;
	MStatus jumpToElement(MArrayDataHandle& hArray, unsigned int index);

	static  void*		creator();
	static  MStatus		initialize();

public:
	
	static	MObject		deviceType;			// GPU or CPU.
	static	MObject		modelFilePath;		// The trained model path.
	static	MObject		inDataFilePath;
	static	MObject		inputs;

	static	MObject		matrix;
	static	MObject		outputs;

	static	ModelCache	modelCache;			// Cache for loading model only once.
	static  bool		_debug;
	static	MTypeId		id;
};
