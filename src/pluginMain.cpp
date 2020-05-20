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