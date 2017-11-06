#include "GeneralizedSparseDRBMTrainer.h"
#include "GeneralizedSparseDRBM.h"

GeneralizedSparseDRBMTrainer::GeneralizedSparseDRBMTrainer()
{
}

GeneralizedSparseDRBMTrainer::GeneralizedSparseDRBMTrainer(GeneralizedSparseDRBM & drbm)
{
	this->gradient.biasH.setConstant(drbm.hSize, 0.0);
	this->gradient.biasY.setConstant(drbm.ySize, 0.0);
	this->gradient.sparseH.setConstant(drbm.hSize, 0.0);
	this->gradient.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->gradient.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->optimizer = GeneralizedSparseDRBMOptimizer(drbm);
}


GeneralizedSparseDRBMTrainer::~GeneralizedSparseDRBMTrainer()
{
}

void GeneralizedSparseDRBMTrainer::train(GeneralizedSparseDRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes)
{
	//(SGD)
	this->gradient.biasH.setConstant(0.0);
	this->gradient.biasY.setConstant(0.0);
	this->gradient.sparseH.setConstant(0.0);
	this->gradient.weightXH.setConstant(0.0);
	this->gradient.weightHY.setConstant(0.0);
	double inv_batch_size = 1.0 / batch_indexes.size();

	for (auto & index : batch_indexes) {
		auto & data = dataset[index];
		drbm.nodeX = data;
		auto & label = labelset[index];

		auto z = drbm.normalizeConstant();

		// Gradient
		auto mujk = drbm.muJKMatrix();

		for (auto i = 0; i < drbm.xSize; i++) {
			for (auto j = 0; j < drbm.hSize; j++) {
				auto gradient = this->dataMeanXH(drbm, data, label, i, j, mujk) - drbm.expectedValueXH(i, j, z, mujk);
				this->gradient.weightXH(i, j) += gradient * inv_batch_size;
			}
		}
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->dataMeanH(drbm, data, label, j, mujk) - drbm.expectedValueH(j, z, mujk);
			this->gradient.biasH(j) += gradient * inv_batch_size;
		}
		for (auto j = 0; j < drbm.hSize; j++) {
			for (auto k = 0; k < drbm.ySize; k++) {
				auto gradient = this->dataMeanHY(drbm, data, label, j, k, mujk) - drbm.expectedValueHY(j, k, z, mujk);
				this->gradient.weightHY(j, k) += gradient * inv_batch_size;
			}
		}
		for (auto k = 0; k < drbm.ySize; k++) {
			auto gradient = this->dataMeanY(drbm, data, label, k, mujk) - drbm.expectedValueY(k, z, mujk);
			this->gradient.biasY(k) += gradient * inv_batch_size;
		}
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->dataMeanAbsHExpSparse(drbm, data, label, j, mujk) - drbm.expectedValueAbsHExpSparse(j, z, mujk);
			this->gradient.sparseH(j) += gradient * inv_batch_size;
		}
	}

	// update
	for (auto i = 0; i < drbm.xSize; i++) {
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->gradient.weightXH(i, j);
			auto delta = this->optimizer.deltaWeightXH(i, j, gradient);
			auto new_param = drbm.weightXH(i, j) + delta;
			drbm.weightXH(i, j) = new_param;
			//drbm.weightXH(i, j) += gradient * 0.01;
		}
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		auto gradient = this->gradient.biasH(j);
		auto delta = this->optimizer.deltaBiasH(j, gradient);
		auto new_param = drbm.biasH(j) + delta;
		drbm.biasH(j) = new_param;
		//drbm.biasH(j) += gradient * 0.01;
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		for (auto k = 0; k < drbm.ySize; k++) {
			auto gradient = this->gradient.weightHY(j, k);
			auto delta = this->optimizer.deltaWeightHY(j, k, gradient);
			auto new_param = drbm.weightHY(j, k) + delta;
			drbm.weightHY(j, k) = new_param;
			//drbm.weightHY(j, k) += gradient * 0.01;
		}
	}
	for (auto k = 0; k < drbm.ySize; k++) {
		auto gradient = this->gradient.biasY(k);
		auto delta = this->optimizer.deltaBiasY(k, gradient);
		auto new_param = drbm.biasY(k) + delta;
		drbm.biasY(k) = new_param;
		//	//drbm.biasY(k) += gradient * 0.01;
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		auto gradient = this->gradient.sparseH(j);
		auto delta = this->optimizer.deltaSparseH(j, gradient);
		auto new_param = drbm.sparseH(j) + delta;
		drbm.sparseH(j) = new_param;
		//	//drbm.sparseH(j) += gradient * 0.01;
	}

	// update optimizer
	this->optimizer.iteration++;

}

double GeneralizedSparseDRBMTrainer::dataMeanXH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex)
{
	//double z_j = drbm.miniNormalizeConstantHidden(hindex);
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//double sum_h_j = 0.0;
	//for (auto & h_j : drbm.hiddenValueSet) {
	//	sum_h_j += h_j * exp(mu * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	//}
	//auto value = data(xindex) * sum_h_j;
	//return value / z_j;
	return 0;
}

double GeneralizedSparseDRBMTrainer::dataMeanXH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	double z_j = drbm.miniNormalizeConstantHidden(hindex);
	double sum_h_j = 0.0;
	for (auto & h_j : drbm.hiddenValueSet) {
		sum_h_j += h_j * exp(mujk(hindex) * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	}
	auto value = data(xindex) * sum_h_j / z_j;
	return value;
}

double GeneralizedSparseDRBMTrainer::dataMeanH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}

	//double z_j = drbm.miniNormalizeConstantHidden(hindex);
	//double sum_h_j = 0.0;
	//for (auto & h_j : drbm.hiddenValueSet) {
	//	sum_h_j += h_j * exp(mu * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	//}
	//auto value = sum_h_j / z_j;
	//return value;
	return 0;
}

double GeneralizedSparseDRBMTrainer::dataMeanH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	double z_j = drbm.miniNormalizeConstantHidden(hindex);
	double sum_h_j = 0.0;
	for (auto & h_j : drbm.hiddenValueSet) {
		sum_h_j += h_j * exp(mujk(hindex, label) * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	}
	auto value = sum_h_j / z_j;
	return value;
}

double GeneralizedSparseDRBMTrainer::dataMeanHY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex)
{
	//if (yindex != label) return 0.0;
	//double z_j = drbm.miniNormalizeConstantHidden(hindex);
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}

	//double sum_h_j = 0.0;
	//for (auto & h_j : drbm.hiddenValueSet) {
	//	sum_h_j += h_j * exp(mu * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	//}
	//auto value = sum_h_j / z_j;
	//return value;
	return 0;
}

double GeneralizedSparseDRBMTrainer::dataMeanHY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & mujk)
{
	if (yindex != label) return 0.0;

	double z_j = drbm.miniNormalizeConstantHidden(hindex);
	double sum_h_j = 0.0;
	for (auto & h_j : drbm.hiddenValueSet) {
		sum_h_j += h_j * exp(mujk(hindex, label) * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	}
	auto value = sum_h_j / z_j;
	return value;
}

double GeneralizedSparseDRBMTrainer::dataMeanY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}

double GeneralizedSparseDRBMTrainer::dataMeanY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}

double GeneralizedSparseDRBMTrainer::dataMeanAbsHExpSparse(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex)
{
	auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	for (auto i = 0; i < drbm.xSize; i++) {
		mu += drbm.weightXH(i, hindex) * data[i];
	}

	double z_j = drbm.miniNormalizeConstantHidden(hindex);

	double sum_h_j = 0.0;
	for (auto & h_j : drbm.hiddenValueSet) {
		sum_h_j += -exp(drbm.sparseH(hindex)) * abs(h_j) * exp(mu * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	}
	auto value = sum_h_j / z_j;
	return value;
	}

double GeneralizedSparseDRBMTrainer::dataMeanAbsHExpSparse(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk)
{
	double z_j = drbm.miniNormalizeConstantHidden(hindex);
	double sum_h_j = 0.0;
	for (auto & h_j : drbm.hiddenValueSet) {
		sum_h_j += -exp(drbm.sparseH(hindex)) * abs(h_j) * exp(mujk(hindex, label) * h_j - exp(drbm.sparseH(hindex)) * abs(h_j));
	}
	auto value = sum_h_j / z_j;
	return value;
}
