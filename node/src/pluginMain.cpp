#include "ml_Rivet.h"
#include <maya/MStreamUtils.h>
#include <maya/MFnPlugin.h>

MStatus initializePlugin( MObject obj )
{
    MStatus status;
	std::cout.set_rdbuf(MStreamUtils::stdOutStream().rdbuf());
	std::cerr.set_rdbuf(MStreamUtils::stdErrorStream().rdbuf());

    MFnPlugin fnPlugin( obj, "None", "1.1", "Any" );

    status = fnPlugin.registerNode( "ml_rivet",
        mlRivet::id,
        mlRivet::creator,
        mlRivet::initialize );
    CHECK_MSTATUS_AND_RETURN_IT( status );
    return MS::kSuccess;
}


MStatus uninitializePlugin( MObject obj )
{
    MStatus status;

    MFnPlugin fnPlugin( obj );

    status = fnPlugin.deregisterNode( mlRivet::id );
    CHECK_MSTATUS_AND_RETURN_IT( status );

    return MS::kSuccess;
}