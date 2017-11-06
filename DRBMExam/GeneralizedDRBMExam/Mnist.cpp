#include "Mnist.h"
#include <fstream>



Mnist::Mnist() {
    throw;
}


Mnist::~Mnist() {
}

Mnist::Mnist(std::string & datafile, std::string & labelfile) {
    this->readData(datafile);
    this->readLabel(labelfile);
    this->normalizeData();
}

void Mnist::readData(std::string & datafile) {
    std::ifstream ifs(datafile, std::ios::binary | std::ios::in);

    int magic_number = 0;
    int image_num = 0;
    int row = 0;
    int col = 0;


    ifs.read(reinterpret_cast<char *>(&magic_number), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&image_num), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&row), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&col), sizeof(int));

    magic_number = this->reverseByteOrder(magic_number);
    image_num = this->reverseByteOrder(image_num);
    row = this->reverseByteOrder(row);
    col = this->reverseByteOrder(col);

    std::vector< std::vector<double> > dataset(image_num);
    for (auto & data : dataset) data = std::vector<double>(row * col);

    for (int n = 0; n < image_num; n++) {
        auto pixel_size = row * row;
        for (int i = 0; i < pixel_size; i++) {
            unsigned char pixel;
            ifs.read(reinterpret_cast<char *>(&pixel), sizeof(unsigned char));
            dataset[n][i] = static_cast<double>(pixel);
        }
    }

    this->dataset = dataset;
}

void Mnist::readLabel(std::string & labelfile) {
    std::ifstream ifs(labelfile, std::ios::binary | std::ios::in);

    int magic_number = 0;
    int image_num = 0;
    int row = 0;
    int col = 0;

    ifs.read(reinterpret_cast<char *>(&magic_number), sizeof(int));
    ifs.read(reinterpret_cast<char *>(&image_num), sizeof(int));

    magic_number = this->reverseByteOrder(magic_number);
    image_num = this->reverseByteOrder(image_num);

    std::vector<int> labelset(image_num);

    for (int n = 0; n < image_num; n++) {
        unsigned char label;
        ifs.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));

        labelset[n] = static_cast<int>(label);
    }

    this->labelset = labelset;
}

void Mnist::normalizeData() {
    for (auto & data : this->dataset) {
        for (auto & pixel : data) {
            pixel *= (1.0 / 255.0);
        }
    }
}

int Mnist::reverseByteOrder(int i) {
    // {c1, c2, c3, c4} ->{c4, c3, c2, c1}
    unsigned char c1, c2, c3, c4;

    c1 = (i >> 24) & 0xFF;
    c2 = (i >> 16) & 0xFF;
    c3 = (i >> 8) & 0xFF;
    c4 = i & 0xFF;

    return (static_cast<int>(c4) << 24) + (static_cast<int>(c3) << 16) + (static_cast<int>(c2) << 8) + static_cast<int>(c1);
}

