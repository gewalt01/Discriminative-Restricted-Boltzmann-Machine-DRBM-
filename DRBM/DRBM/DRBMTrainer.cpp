#include "DRBMTrainer.h"
#include "DRBM.h"



DRBMTrainer::DRBMTrainer()
{
}

DRBMTrainer::DRBMTrainer(DRBM & drbm)
{
	this->gradient.biasC.setConstant(drbm.hSize, 0.0);
	this->gradient.biasD.setConstant(drbm.ySize, 0.0);
	this->gradient.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->gradient.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->optimizer = DRBMOptimizer(drbm);
}


DRBMTrainer::~DRBMTrainer()
{
}

void DRBMTrainer::train(DRBM & drbm, Eigen::VectorXf & data, int label)
{
	// Online Learning(SGD)

	auto z = drbm.normalizeConstantDiv2H();

	// Gradient
	for (auto i = 0; i < drbm.xSize; i++) {
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->dataMeanXH(drbm, data, label, i, j) - drbm.expectedValueXH(i, j, z);
			this->gradient.weightXH(i, j) = gradient;
		}
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		auto gradient = this->dataMeanH(drbm, data, label, j) - drbm.expectedValueH(j, z);
		this->gradient.biasC(j) = gradient;
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		for (auto k = 0; k < drbm.ySize; k++) {
			auto gradient = this->dataMeanHY(drbm, data, label, j, k) - drbm.expectedValueHY(j, k, z);
			this->gradient.weightHY(j, k) = gradient;
		}
	}
	for (auto k = 0; k < drbm.ySize; k++) {
		auto gradient = this->dataMeanY(drbm, data, label, k) - drbm.expectedValueY(k, z);
		this->gradient.biasD(k) = gradient;
	}

	// update
	for (auto i = 0; i < drbm.xSize; i++) {
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->gradient.weightXH(i, j);
			auto delta = this->optimizer.deltaWeightXH(i, j, gradient);
			auto new_param = drbm.weightXH(i, j) + delta;
			drbm.weightXH(i, j) = new_param;
		}
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		auto gradient = this->gradient.biasC(j);
		auto delta = this->optimizer.deltaBiasC(j, gradient);
		auto new_param = drbm.biasC(j) + delta;
		drbm.biasC(j) = new_param;
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		for (auto k = 0; k < drbm.ySize; k++) {
			auto gradient = this->gradient.weightHY(j, k);
			auto delta = this->optimizer.deltaWeightHY(j, k, gradient);
			auto new_param = drbm.weightHY(j, k) + delta;
			drbm.weightHY(j, k) = new_param;
		}
	}
	for (auto k = 0; k < drbm.ySize; k++) {
		auto gradient = this->gradient.biasD(k);
		auto delta = this->optimizer.deltaBiasD(k, gradient);
		auto new_param = drbm.biasD(k) + delta;
		drbm.biasD(k) = new_param;
	}

	// update optimizer
	this->optimizer.iteration++;

}

double DRBMTrainer::dataMeanXH(DRBM & drbm, Eigen::VectorXf & data, int label, int xindex, int hindex)
{
	auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	for (auto i = 0; i < drbm.xSize; i++) {
		mu += drbm.weightXH(i, hindex) * data[i];
	}
	auto value = data.x[xindex] * tanh(mu);
	return value;
}

double DRBMTrainer::dataMeanH(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex)
{
	auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	for (auto i = 0; i < drbm.xSize; i++) {
		mu += drbm.weightXH(i, hindex) * data[i];
	}
	auto value = tanh(mu);
	return value;
}

double DRBMTrainer::dataMeanHY(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, int yindex)
{
	if (yindex != label) return 0.0;
	auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	for (auto i = 0; i < drbm.xSize; i++) {
		mu += drbm.weightXH(i, hindex) * data[i];
	}
	auto value = tanh(mu);
	return value;
}

double DRBMTrainer::dataMeanY(DRBM & drbm, Eigen::VectorXf & data, int label, int yindex)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}
