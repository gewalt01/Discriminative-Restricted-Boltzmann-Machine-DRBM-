#include "SparseDRBM.h"
#include <vector>
#include <numeric>
#include <algorithm>

SparseDRBM::SparseDRBM()
{
}

SparseDRBM::SparseDRBM(size_t xsize, size_t hsize, size_t ysize)
{
	this->xSize = xsize;
	this->hSize = hsize;
	this->ySize = ysize;

	this->nodeX.setConstant(xsize, 0.0);
	this->nodeH.setConstant(hsize, 0.0);
	this->nodeY.setConstant(ysize, 0.0);
	this->biasH.setConstant(hsize, 0.0);
	this->biasY.setConstant(ysize, 0.0);
	this->sparseH.setConstant(hsize, 0.0);
	this->weightXH.setRandom(xsize, hsize) /= sqrt(xsize + ysize);
	this->weightHY.setRandom(hsize, ysize) /= sqrt(hsize);
}


SparseDRBM::~SparseDRBM()
{
}

double SparseDRBM::normalizeConstant()
{
	auto value = pow(2, this->hSize) * this->normalizeConstantDiv2H();

	return value;
}

double SparseDRBM::normalizeConstantDiv2H()
{
	auto value = 0.0;
	for (int k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (int j = 0; j < this->hSize; j++) {
			k_val *=  exp( - exp( this->sparseH(j) ) ) * cosh(this->muJK(j, k));
		}
		value += k_val;
	}

	return value;
}

double SparseDRBM::muJK(int hindex, int yindex)
{
	// XXX: Yノードに値が適切にセットされている必要がある
	auto value = this->biasH(hindex) + this->weightHY(hindex, yindex);
	value += this->weightXH.col(hindex).dot(this->nodeX);
	//for (int i = 0; i < this->xSize; i++) {
	//	value += this->weightXH(i, hindex) * this->nodeX(i);
	//}

	return value;
}

Eigen::MatrixXd SparseDRBM::muJKMatrix()
{
	// XXX: Yノードに値が適切にセットされている必要がある
	Eigen::MatrixXd mujk(this->hSize, this->ySize);
	for (int j = 0; j < this->hSize; j++) {
		for (int k = 0; k < this->ySize; k++) {
			mujk(j, k) = this->muJK(j, k);
		}
	}

	return mujk;
}

double SparseDRBM::condProbY(int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->condProbY(yindex, z);

	return value;
}

double SparseDRBM::condProbY(int yindex, double z)
{
	auto & z_k = z;
	auto potential = 0.0; {
		auto k_val = exp(this->biasY(yindex));
		for (auto j = 0; j < this->hSize; j++) {
			auto mu_jk = this->muJK(j, yindex);
			k_val += exp( - exp( this->sparseH(j) ) ) * cosh(mu_jk);
		}
		potential += k_val;
	}
	auto value = potential / z_k;
	return value;
}

int SparseDRBM::maxCondProbYIndex()
{
	std::vector<double> probs(this->ySize);
	auto z_k = this->normalizeConstantDiv2H();
	for (int k = 0; k < this->ySize; k++) {
		probs[k] = condProbY(k, z_k);
	}

	auto max_itr = std::max_element(probs.begin(), probs.end());
	auto index = std::distance(probs.begin(), max_itr);

	return index;

}

double SparseDRBM::expectedValueXH(int xindex, int hindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueXH(xindex, hindex, z);

	return value;
}

double SparseDRBM::expectedValueXH(int xindex, int hindex, double z)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z);

	return value;
}

double SparseDRBM::expectedValueXH(int xindex, int hindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z, mujk);

	return value;
}

double SparseDRBM::expectedValueH(int hindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueH(hindex, z);

	return value;
}

double SparseDRBM::expectedValueH(int hindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= exp( - exp( this->sparseH(l) ) ) * cosh(this->muJK(l, k));
		}
		k_val *= exp(-exp(this->sparseH(hindex))) *  sinh(this->muJK(hindex, k));
		value += k_val;
	}
	value /= z;

	return value;
}

double SparseDRBM::expectedValueH(int hindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= exp(-exp(this->sparseH(l))) *  cosh(mujk(l, k));
		}
		k_val *= exp(-exp(this->sparseH(hindex))) *  sinh(mujk(hindex, k));
		value += k_val;
	}
	value = value / z;

	return value;
}

double SparseDRBM::expectedValueHY(int hindex, int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueHY(hindex, yindex, z);

	return value;
}

double SparseDRBM::expectedValueHY(int hindex, int yindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (auto & l : lindex) {
		value *= exp(-exp(this->sparseH(l))) *  cosh(this->muJK(l, yindex));
	}

	value *= exp(-exp(this->sparseH(hindex))) *  sinh(this->muJK(hindex, yindex));
	value = value / z;

	return value;
}

double SparseDRBM::expectedValueHY(int hindex, int yindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	for (auto & l : lindex) {
		value *= exp(-exp(this->sparseH(l))) * cosh(mujk(l, yindex));
	}

	value *= exp(-exp(this->sparseH(hindex))) *  sinh(mujk(hindex, yindex));
	value = value / z;

	return value;
}

double SparseDRBM::expectedValueY(int yindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueY(yindex, z);

	return value;
}

double SparseDRBM::expectedValueY(int yindex, double z)
{
	auto value = exp(this->biasY(yindex));
	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (int j = 0; j < this->hSize; j++) {
		value *= exp(-exp(this->sparseH(j))) *  cosh(this->muJK(j, yindex));
	}

	value = value / z;

	return value;
}

double SparseDRBM::expectedValueY(int yindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = exp(this->biasY(yindex));
	for (int j = 0; j < this->hSize; j++) {
		value *= exp(-exp(this->sparseH(j))) *  cosh(mujk(j, yindex));
	}

	value = value / z;

	return value;
}

double SparseDRBM::expectedValueAbsHExpSparse(int hindex)
{
	auto z = this->normalizeConstantDiv2H();
	auto value = this->expectedValueAbsHExpSparse(hindex, z);

	return value;
}

double SparseDRBM::expectedValueAbsHExpSparse(int hindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= exp(-exp(this->sparseH(l))) * cosh(this->muJK(l, k));
		}
		k_val *= -exp(this->sparseH(hindex)) * exp(-exp(this->sparseH(hindex))) *  cosh(this->muJK(hindex, k));
		value += k_val;
	}
	value /= z;

	return value;
}

double SparseDRBM::expectedValueAbsHExpSparse(int hindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= exp(-exp(this->sparseH(l))) *  cosh(mujk(l, k));
		}
		k_val *= -exp(this->sparseH(hindex)) * exp(-exp(this->sparseH(hindex))) *  cosh(this->muJK(hindex, k));
		value += k_val;
	}
	value = value / z;

	return value;
}
