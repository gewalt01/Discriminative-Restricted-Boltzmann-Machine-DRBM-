#include "DRBM.h"
#include <vector>
#include <numeric>
#include <algorithm>

DRBM::DRBM()
{
}

DRBM::DRBM(size_t xsize, size_t hsize, size_t ysize)
{
	this->xSize = xsize;
	this->hSize = hsize;
	this->ySize = ysize;

	this->nodeX.setConstant(xsize, 0.0);
	this->nodeH.setConstant(xsize, 0.0);
	this->nodeY.setConstant(xsize, 0.0);
	this->biasC.setRandom(hsize) * 0.01;
	this->biasD.setRandom(ysize) * 0.01;
	this->weightXH(xsize, hsize) * 0.01;
	this->weightHY(hsize, ysize) * 0.01;
}


DRBM::~DRBM()
{
}

double DRBM::normalizeConstant()
{
	auto value = pow(2, this->hSize) * this->normalizeConstantDiv2H();

	return value;
}

double DRBM::normalizeConstantDiv2H()
{
	auto value = 0.0;
	for (int k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasD(k));
		for (int j = 0; j < this->hSize; j++) {
			k_val += cosh(this->muJK(j, k));
		}
		value += k_val;
	}

	return value;
}

double DRBM::muJK(int hindex, int yindex)
{
	auto value = this->biasC(hindex) + this->weightHY(hindex, yindex);
	for (int i = 0; i < this->xSize; i++) {
		value += this->weightXH(i, hindex) * this->nodeX(i);
	}

	return value;
}

double DRBM::condProbY(int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->condProbY(yindex, z);

	return value;
}

double DRBM::condProbY(int yindex, double z)
{
	auto z_k = this->normalizeConstantDiv2H();
	auto potential = 0.0; {
		auto k_val = exp(this->biasD(yindex));
		for (auto j = 0; j < this->hSize; j++) {
			auto mu_j = this->biasC(j) + this->weightHY(j, yindex);
			for (auto i = 0; i < this->xSize; i++) {
				mu_j += this->weightXH(i, j);
			}
			k_val += cosh(mu_j);
		}
		potential += k_val;
	}
	auto value = potential / z_k;
	return value;
}

double DRBM::expectedValueXH(int xindex, int hindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueXH(xindex, hindex, z);

	return value;
}

double DRBM::expectedValueXH(int xindex, int hindex, double z)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z);

	return value;
}

double DRBM::expectedValueH(int hindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueH(hindex, z);

	return value;
}

double DRBM::expectedValueH(int hindex, double z)
{
	std::vector<int> lindex(10);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasD(k));
		for (auto & l : lindex) {
			k_val *= cosh(this->muJK(l, k));
		}
		k_val *= sinh(this->muJK(hindex, k));
		value += k_val;
	}
	value /= z;

	return value;
}

double DRBM::expectedValueHY(int hindex, int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueHY(hindex, yindex, z);

	return value;
}

double DRBM::expectedValueHY(int hindex, int yindex, double z)
{
	std::vector<int> lindex(10);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasD(yindex));

	for (auto & l : lindex) {
		value *= cosh(this->muJK(l, yindex));
	}

	value *= sinh(this->muJK(hindex, yindex));
	value /= z;

	return value;
}

double DRBM::expectedValueY(int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueY(yindex, z);

	return value;
}

double DRBM::expectedValueY(int yindex, double z)
{
	auto value = exp(this->biasD(yindex));
	for (int j = 0; j < this->hSize; j++) {
		value *= cosh(this->muJK(j, yindex));
	}

	value /= z;

	return value;
}
