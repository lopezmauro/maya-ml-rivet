/*MIT License

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
#include<iostream>
#include<string>
#include <zip.h>
#include <cstdlib>
#include <valarray> // for value array operations 
#include <vector> 
#include <sstream> 
#include <fstream>
#include <map>


#pragma warning(disable : 4996)
char *readfile(std::string filename, size_t *size) {
	char * buffer;
	size_t result;

	FILE* pFile = fopen(filename.c_str(), "rb");
	if (pFile == NULL) { fputs("File error", stderr); exit(1); }

	// obtain file size:
	fseek(pFile, 0, SEEK_END);
	unsigned int lSize = ftell(pFile);
	rewind(pFile);

	// allocate memory to contain the whole file:
	buffer = (char*)malloc(sizeof(char)*lSize);
	if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

	// copy the file into the buffer:
	result = fread(buffer, 1, lSize, pFile);
	if (result != lSize) { fputs("Reading error", stderr); exit(3); }

	/* the whole file is now loaded in the memory buffer. */

	// terminate
	fclose(pFile);
	*size = lSize;
	return buffer;

}
bool hasEnding(std::string const &fullString, std::string const &ending) {
	if (fullString.length() >= ending.length()) {
		return (0 == fullString.compare(fullString.length() - ending.length(), ending.length(), ending));
	}
	else {
		return false;
	}
}

std::vector<float> readCsvFile(char *buffer)
{

	std::stringstream ss(buffer);
	std::string to;
	std::vector<float> v;

	if (buffer != NULL)
	{
		while (std::getline(ss, to, '\n')) {
			float f = std::stof(to);
			v.push_back(f);
		}
	}
	return v;
}

void dumpCsv(zip *archive, std::map<std::string, std::valarray<float>> &fileMap) {
	int files_total = zip_get_num_entries(archive, 0);
	if (files_total == 0) 
	{
		return;
	}
	struct zip_stat sb;
	int r, len;
	long long sum;
	char buffer[100000];
	std::string csvFormat = ".csv";
		
	for (int i = 0; i < files_total; i++) {
		if (zip_stat_index(archive, i, 0, &sb) == 0) {

			zip_file *zf = zip_fopen_index(archive, i, 0);
			if (!zf) {
				std::cout << "failed to open  entry of archive. " << zip_strerror(archive) << std::endl;
				zip_close(archive);
			}

			if (hasEnding(sb.name, csvFormat))
			{
				std::cout << "Reading csv file "<< sb.name << " size " << sb.size << std::endl;
			}
			sum = 0;
			zip_fread(zf, buffer, sb.size);
			zip_fclose(zf);

			std::vector<float> val = readCsvFile(buffer);
			std::valarray<float> result(val.data(), val.size());
			fileMap[sb.name] = result;
		}
	}
}


zip * get_archive(std::string path, int flags) {
	int error = 0;
	zip *archive = zip_open(path.c_str(), flags, &error);

	if (!archive) {
		std::cout << "could not open or create archive" << path << std::endl;
		exit(1);
	}
	std::cout << "Done : creating archieve" << path << std::endl;
	return archive;
}

void printVector(std::string name, std::valarray <float> const &a) {
	std::cout << name << " : [";
	for (int i = 0; i < a.size(); i++)
		std::cout << a[i] << ", ";
	std::cout << "]" << endl;
}


int main() {

	std::string path = "D:/projects/ml_rivet/model/model.zip";

	//Read
	zip *archive1 = get_archive(path, ZIP_CREATE);
	std::map<std::string, std::valarray<float>> fileMap;
	dumpCsv(archive1, fileMap);
	zip_close(archive1);

	std::map<std::string, std::valarray<float>>::iterator it = fileMap.begin();

	// Iterate over the map using Iterator till end.
	while (it != fileMap.end())
	{
		// Accessing KEY from element pointed by it.
		std::string name = it->first;

		// Accessing VALUE from element pointed by it.
		std::valarray<float> value = it->second;

		printVector(name, value);
		// Increment the Iterator to point to next entry
		it++;
	}

	return 0;
}