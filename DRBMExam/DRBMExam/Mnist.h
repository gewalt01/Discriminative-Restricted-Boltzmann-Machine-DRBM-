#pragma once

#include <string>
#include <vector>

class Mnist
{
public:
    std::vector< std::vector<double> > dataset;
    std::vector< int > labelset;

public:
    Mnist();
    Mnist(std::string & datafile, std::string & labelfile);
    ~Mnist();

    void readData(std::string & datafile);
    void readLabel(std::string & labelfile);
    int reverseByteOrder(int i);
    void normalizeData();
};

