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
	drbm.nodeX = data;

	auto z = drbm.normalizeConstantDiv2H();

	// Gradient
	auto mujk = drbm.muJKMatrix();

	for (auto i = 0; i < drbm.xSize; i++) {
		for (auto j = 0; j < drbm.hSize; j++) {
			auto gradient = this->dataMeanXH(drbm, data, label, i, j, mujk) - drbm.expectedValueXH(i, j, z, mujk);
			this->gradient.weightXH(i, j) = gradient;
		}
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		auto gradient = this->dataMeanH(drbm, data, label, j, mujk) - drbm.expectedValueH(j, z, mujk);
		this->gradient.biasC(j) = gradient;
	}
	for (auto j = 0; j < drbm.hSize; j++) {
		for (auto k = 0; k < drbm.ySize; k++) {
			auto gradient = this->dataMeanHY(drbm, data, label, j, k, mujk) - drbm.expectedValueHY(j, k, z, mujk);
			this->gradient.weightHY(j, k) = gradient;
		}
	}
	for (auto k = 0; k < drbm.ySize; k++) {
		auto gradient = this->dataMeanY(drbm, data, label, k, mujk) - drbm.expectedValueY(k, z, mujk);
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
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = data(xindex) * tanh(mu);
	//return value;
}

double DRBMTrainer::dataMeanXH(DRBM & drbm, Eigen::VectorXf & data, int label, int xindex, int hindex, Eigen::MatrixXf & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	auto value = data(xindex) * tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanH(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = tanh(mu);
	//return value;
}

double DRBMTrainer::dataMeanH(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, Eigen::MatrixXf & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	auto value = tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanHY(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, int yindex)
{
	//if (yindex != label) return 0.0;
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasC(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = tanh(mu);
	//return value;
}

double DRBMTrainer::dataMeanHY(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, int yindex, Eigen::MatrixXf & mujk)
{
	if (yindex != label) return 0.0;
	// FIXME: muの計算使いまわしできそうだけど…
	auto value = tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanY(DRBM & drbm, Eigen::VectorXf & data, int label, int yindex)
{
	//auto value = (yindex != label) ? 0.0 : 1.0;
	//return value;
}

double DRBMTrainer::dataMeanY(DRBM & drbm, Eigen::VectorXf & data, int label, int yindex, Eigen::MatrixXf & muJK)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}
