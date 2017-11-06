#include "GeneralizedSparseDRBM.h"
#include <vector>
#include <numeric>
#include <algorithm>

GeneralizedSparseDRBM::GeneralizedSparseDRBM()
{
}

GeneralizedSparseDRBM::GeneralizedSparseDRBM(size_t xsize, size_t hsize, size_t ysize)
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


GeneralizedSparseDRBM::~GeneralizedSparseDRBM()
{
}

double GeneralizedSparseDRBM::normalizeConstant()
{
	auto value = 0.0;
	auto mujk = this->muJKMatrix();

	for (int k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (int j = 0; j < this->hSize; j++) {
			double sum_h_j = 0.0;
			for (auto & h_j : this->hiddenValueSet) {
				sum_h_j += exp(mujk(j, k) * h_j - exp(this->sparseH(j)) * abs(h_j));
			}
			k_val *= sum_h_j;
		}
		value += k_val;
	}

	return value;
}

double GeneralizedSparseDRBM::muJ(int hindex)
{
	// XXX: Yノードに値が適切にセットされている必要がある
	auto value = this->biasH(hindex);
	value += this->weightXH.col(hindex).dot(this->nodeX);
	for (int k = 0; k < this->ySize; k++) {
		value += this->weightHY(hindex, k) * this->nodeY(k);
	}
	//for (int i = 0; i < this->xSize; i++) {
	//	value += this->weightXH(i, hindex) * this->nodeX(i);
	//}

	return value;
}

double GeneralizedSparseDRBM::muJK(int hindex, int yindex)
{
	// XXX: Yノードに値が適切にセットされている必要がある
	auto value = this->biasH(hindex) + this->weightHY(hindex, yindex);
	value += this->weightXH.col(hindex).dot(this->nodeX);
	//for (int i = 0; i < this->xSize; i++) {
	//	value += this->weightXH(i, hindex) * this->nodeX(i);
	//}

	return value;
}

Eigen::MatrixXd GeneralizedSparseDRBM::muJKMatrix()
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

double GeneralizedSparseDRBM::condProbY(int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->condProbY(yindex, z);

	return value;
}

double GeneralizedSparseDRBM::condProbY(int yindex, double z)
{
	auto & z_k = z;
	auto potential = 0.0; {
		auto k_val = exp(this->biasY(yindex));
		for (auto j = 0; j < this->hSize; j++) {
			auto mu_jk = this->muJK(j, yindex);
			double sum_h_j = 0.0;
			for (auto & h_j : this->hiddenValueSet) {
				sum_h_j += exp(mu_jk * h_j - exp(this->sparseH(j)) * abs(h_j));
			}
			k_val += sum_h_j;
		}
		potential += k_val;
	}
	auto value = potential / z_k;
	return value;
}

int GeneralizedSparseDRBM::maxCondProbYIndex()
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

double GeneralizedSparseDRBM::expectedValueXH(int xindex, int hindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueXH(xindex, hindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueXH(int xindex, int hindex, double z)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueXH(int xindex, int hindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = this->nodeX(xindex) * this->expectedValueH(hindex, z, mujk);

	return value;
}

double GeneralizedSparseDRBM::expectedValueH(int hindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueH(hindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueH(int hindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			double mu_lk = this->muJK(l, k);
			double sum_h_l = 0.0;
			for (auto & h_l : this->hiddenValueSet) {
				sum_h_l += exp(mu_lk * h_l - exp(this->sparseH(hindex) * abs(h_l)));
			}
			k_val *= sum_h_l;
		}

		double mu_jk = this->muJK(hindex, k);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += h_j * exp(mu_jk * h_j - exp(this->sparseH(hindex) * abs(h_j)));
		}
		k_val *= sum_h_j;
		value += k_val;
	}
	value /= z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueH(int hindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			double mu_lk = mujk(l, k);
			double sum_h_l = 0.0;
			for (auto & h_l : this->hiddenValueSet) {
				sum_h_l += exp(mu_lk * h_l - exp(this->sparseH(hindex) * abs(h_l)));
			}
			k_val *= sum_h_l;
		}

		double mu_jk = mujk(hindex, k);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += h_j * exp(mu_jk * h_j - exp(this->sparseH(hindex) * abs(h_j)));
		}
		k_val *= sum_h_j;
		value += k_val;
	}
	value /= z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueHY(int hindex, int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueHY(hindex, yindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueHY(int hindex, int yindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (auto & l : lindex) {
		double mu_lk = this->muJK(l, yindex);
		double sum_h_l = 0.0;;
		for (auto & h_l : this->hiddenValueSet) {
			sum_h_l += exp(mu_lk * h_l - exp(this->sparseH(hindex)) * abs(h_l));
		}
		value *= sum_h_l;
	}

	double mu_jk = this->muJK(hindex, yindex);
	double sum_h_j = 0.0;
	for (auto & h_j : this->hiddenValueSet) {
		sum_h_j += h_j *  exp(mu_jk * h_j - exp(this->sparseH(hindex)) * abs(h_j));
	}
	value *= sum_h_j;
	value = value / z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueHY(int hindex, int yindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	auto value = exp(this->biasY(yindex));

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (auto & l : lindex) {
		double mu_lk = mujk(l, yindex);
		double sum_h_l = 0.0;;
		for (auto & h_l : this->hiddenValueSet) {
			sum_h_l += exp(mu_lk * h_l - exp(this->sparseH(hindex)) * abs(h_l));
		}
		value *= sum_h_l;
	}

	double mu_jk = mujk(hindex, yindex);
	double sum_h_j = 0.0;
	for (auto & h_j : this->hiddenValueSet) {
		sum_h_j += h_j *  exp(mu_jk * h_j - exp(this->sparseH(hindex)) * abs(h_j));
	}
	value *= sum_h_j;
	value = value / z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueY(int yindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueY(yindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueY(int yindex, double z)
{
	auto value = exp(this->biasY(yindex));
	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (int j = 0; j < this->hSize; j++) {
		double mu_jk = this->muJK(j, yindex);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += exp(mu_jk * h_j - exp(this->sparseH(j)) * abs(h_j));
		}
		value *= sum_h_j;
	}

	value = value / z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueY(int yindex, double z, Eigen::MatrixXd & mujk)
{
	auto value = exp(this->biasY(yindex));
	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	for (int j = 0; j < this->hSize; j++) {
		double mu_jk = mujk(j, yindex);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += exp(mu_jk * h_j - exp(this->sparseH(j)) * abs(h_j));
		}
		value *= sum_h_j;
	}

	value = value / z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueAbsHExpSparse(int hindex)
{
	auto z = this->normalizeConstant();
	auto value = this->expectedValueAbsHExpSparse(hindex, z);

	return value;
}

double GeneralizedSparseDRBM::expectedValueAbsHExpSparse(int hindex, double z)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			double mu_lk = this->muJK(l, k);
			double sum_h_l = 0.0;
			for (auto & h_l : this->hiddenValueSet) {
				sum_h_l += exp(mu_lk * h_l - exp(sparseH(hindex)) * abs(h_l));
			}
			k_val *= sum_h_l;
		}

		double mu_jk = this->muJK(hindex, k);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += -exp(this->sparseH(hindex)) * abs(h_j) * exp(mu_jk * h_j - exp(this->sparseH(hindex)) * abs(h_j));
		}
		k_val *= sum_h_j;
		value += k_val;
	}
	value /= z;

	return value;
}

double GeneralizedSparseDRBM::expectedValueAbsHExpSparse(int hindex, double z, Eigen::MatrixXd & mujk)
{
	std::vector<int> lindex(this->hSize);
	std::iota(lindex.begin(), lindex.end(), 0);
	lindex.erase(lindex.begin() + hindex);

	// FIXME: muJKの計算を他の期待値計算で使いまわせそうだけども…
	auto value = 0.0;
	for (auto k = 0; k < this->ySize; k++) {
		auto k_val = exp(this->biasY(k));
		for (auto & l : lindex) {
			double mu_lk = mujk(l, k);
			double sum_h_l = 0.0;
			for (auto & h_l : this->hiddenValueSet) {
				sum_h_l += exp(mu_lk * h_l - exp(sparseH(hindex)) * abs(h_l));
			}
			k_val *= sum_h_l;
		}

		double mu_jk = mujk(hindex, k);
		double sum_h_j = 0.0;
		for (auto & h_j : this->hiddenValueSet) {
			sum_h_j += -exp(this->sparseH(hindex)) * abs(h_j) * exp(mu_jk * h_j - exp(this->sparseH(hindex)) * abs(h_j));
		}
		k_val *= sum_h_j;
		value += k_val;
	}
	value /= z;

	return value;
}

double GeneralizedSparseDRBM::miniNormalizeConstantHidden(int hindex)
{
	double sum_h_j = 0.0;
	double mu_j = this->muJ(hindex);
	for (auto & h_j : this->hiddenValueSet) {
		sum_h_j += exp(mu_j * h_j - exp(this->sparseH(hindex)) * abs(h_j));
	}

	return sum_h_j;
}

double GeneralizedSparseDRBM::miniNormalizeConstantHidden(int hindex, int yindex)
{
	double sum_h_j = 0.0;
	double mu_jk = this->muJK(hindex, yindex);
	for (auto & h_j : this->hiddenValueSet) {
		sum_h_j += exp(mu_jk * h_j - exp(this->sparseH(hindex)) * abs(h_j));
	}

	return sum_h_j;
}

std::vector<double> GeneralizedSparseDRBM::splitHiddenSet() {
	std::vector<double> set(divSize + 1);

	auto x = [](double split_size, double i, double min, double max) {  // 分割関数[i=0,1,...,elems]
		return 1.0 / (split_size)* i * (max - min) + min;
	};

	for (int i = 0; i < set.size(); i++) set[i] = x(divSize, i, hMin, hMax);

	return set;
}

int GeneralizedSparseDRBM::getHiddenValueSetSize() {
	return divSize + 1;
}

// 隠れ変数の取りうる最大値を取得
double GeneralizedSparseDRBM::getHiddenMax() {
	return hMax;
}

// 隠れ変数の取りうる最大値を設定
void GeneralizedSparseDRBM::setHiddenMax(double value) {
	hMax = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の取りうる最小値を取得
double GeneralizedSparseDRBM::getHiddenMin() {
	return hMin;
}

// 隠れ変数の取りうる最小値を設定
void GeneralizedSparseDRBM::setHiddenMin(double value) {
	hMin = value;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}

// 隠れ変数の区間分割数を返す
size_t GeneralizedSparseDRBM::getHiddenDivSize() {
	return divSize;
}

// 隠れ変数の区間分割数を設定
void GeneralizedSparseDRBM::setHiddenDivSize(size_t div_size) {
	divSize = div_size;

	// 区間分割
	hiddenValueSet = splitHiddenSet();
}