#include "DRBM01.h"
#include <vector>
#include <numeric>
#include <algorithm>

DRBM01::DRBM01()
{
}

DRBM01::DRBM01(size_t xsize, size_t hsize, size_t ysize)
{
	this->xSize = xsize;
	this->hSize = hsize;
	this->ySize = ysize;

	this->nodeX.setConstant(xsize, 0.0);
	this->nodeH.setConstant(hsize, 0.0);
	this->nodeY.setConstant(ysize, 0.0);
	this->biasH.setConstant(hsize, 0.0);
	this->biasY.setConstant(ysize, 0.0);
	this->weightXH.setRandom(xsize, hsize) /= sqrt(xsize + ysize);
	this->weightHY.setRandom(hsize, ysize) /= sqrt(hsize);
}


DRBM01::~DRBM01()
{
}

double DRBM01::normalizeConstant()
{
	auto value = 0.0;
	auto mu_jk = this->muJKMatrix();
	for (int k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (int j = 0; j < this->hSize; j++) {
			k_val *= (1.0 + exp(mu_jk(j, k)));
		}
		value += k_val;
	}

	return value;
}

double DRBM01::muJK(int hindex, int yindex)
{
	// XXX: Yノードに値が適切にセットされている必要がある
	auto value = this->biasH(hindex) + this->weightHY(hindex, yindex);
	value += this->weightXH.col(hindex).dot(this->nodeX);
	//for (int i = 0; i < this->xSize; i++) {
	//	value += this->weightXH(i, hindex) * this->nodeX(i);
	//}

	return value;
}

Eigen::MatrixXd DRBM01::muJKMatrix()
{
	Eigen::MatrixXd mujk(this->hSize, this->ySize);

	// w^t * x
	auto a = this->weightXH.transpose() * this->nodeX;
	for (int j = 0; j < this->hSize; j++) {
		mujk.row(j).setConstant(a(j) + biasH(j));
	}
	mujk += this->weightHY;

	//for (int j = 0; j < this->hSize; j++) {
	//	for (int k = 0; k < this->ySize; k++) {
	//		mujk(j, k) = this->muJK(j, k);
	//	}
	//}

	return mujk;
}

double DRBM01::condProbY(int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->condProbY(yindex, z);

	return value;
}

double DRBM01::condProbY(int yindex, double z)
{
	auto & z_k = z;
	auto potential = 0.0; {
		auto k_val = exp(this->biasY(yindex));
		for (auto j = 0; j < this->hSize; j++) {
			auto mu_jk = this->muJK(j, yindex);
			k_val += (1.0 + exp(mu_jk));
		}
		potential += k_val;
	}
	auto value = potential / z_k;
	return value;
}

int DRBM01::maxCondProbYIndex()
{
	std::vector<double> probs(this->ySize);
	auto z_k = this->normalizeConstant();
	for (int k = 0; k < this->ySize; k++) {
		probs[k] = condProbY(k, z_k);
	}

	auto max_itr = std::max_element(probs.begin(), probs.end());
	auto index = std::distance(probs.begin(), max_itr);

	return index;

}

double DRBM01::expectedValueXH(int xindex, int hindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueXH(xindex, hindex, z);

	return value;
}

double DRBM01::expectedValueXH(int xindex, int hindex, double z)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z);

	return value;
}

double DRBM01::expectedValueXH(int xindex, int hindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z, mujk);

	return value;
}

double DRBM01::expectedValueH(int hindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueH(hindex, z);

	return value;
}

double DRBM01::expectedValueH(int hindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	auto mu_jk = this->muJKMatrix();
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= (1.0 + exp(mu_jk(l, k)));
		}
		k_val *= exp(mu_jk(hindex, k));
		value += k_val;
	}
	value /= z;

	return value;
}

double DRBM01::expectedValueH(int hindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			k_val *= (1.0 + exp(mujk(l, k)));
		}
		k_val *= exp(mujk(hindex, k));
		value += k_val;
	}
	value = value / z;

	return value;
}

double DRBM01::expectedValueHY(int hindex, int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueHY(hindex, yindex, z);

	return value;
}

double DRBM01::expectedValueHY(int hindex, int yindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (auto & l : lindex) {
		value *= (1.0 + exp(this->muJK(l, yindex)));
	}

	value *= exp(this->muJK(hindex, yindex));
	value = value / z;

	return value;
}

double DRBM01::expectedValueHY(int hindex, int yindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	for (auto & l : lindex) {
		value *= (1.0 + exp(mujk(l, yindex)));
	}

	value *= exp(mujk(hindex, yindex));
	value = value / z;

	return value;
}

double DRBM01::expectedValueY(int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueY(yindex, z);

	return value;
}

double DRBM01::expectedValueY(int yindex, double z)
{
	auto value = exp(this->biasY(yindex));
	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (int j = 0; j < this->hSize; j++) {
		value *= (1.0 + (this->muJK(j, yindex)));
	}

	value = value / z;

	return value;
}

double DRBM01::expectedValueY(int yindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = exp(this->biasY(yindex));
	for (int j = 0; j < this->hSize; j++) {
		value *= (1.0 + exp(mujk(j, yindex)));
	}

	value = value / z;

	return value;
}
