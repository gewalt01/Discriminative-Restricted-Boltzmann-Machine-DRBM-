#include "DRBM01Trainer.h"
#include "DRBM01.h"
#include <omp.h>

DRBM01Trainer::DRBM01Trainer()
{
}

DRBM01Trainer::DRBM01Trainer(DRBM01 & drbm)
{
	this->gradient.biasH.setConstant(drbm.hSize, 0.0);
	this->gradient.biasY.setConstant(drbm.ySize, 0.0);
	this->gradient.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->gradient.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->optimizer = DRBM01Optimizer(drbm);
}


DRBM01Trainer::~DRBM01Trainer()
{
}

void DRBM01Trainer::train(DRBM01 & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes)
{
	//(SGD)
	this->gradient.biasH.setConstant(0.0);
	this->gradient.biasY.setConstant(0.0);
	this->gradient.weightXH.setConstant(0.0);
	this->gradient.weightHY.setConstant(0.0);
	double inv_batch_size = 1.0 / batch_indexes.size();

	auto batch_size = batch_indexes.size();
#pragma omp parallel for
	for (int n = 0; n < batch_size; n++) {
		auto drbm_replica = drbm;
		int index = batch_indexes[n];
		auto & data = dataset[index];
		drbm_replica.nodeX = data;
		auto & label = labelset[index];

		auto mujk = drbm_replica.muJKMatrix();
		auto z = drbm_replica.normalizeConstant(mujk);

		// Gradient
		for (auto i = 0; i < drbm_replica.xSize; i++) {
			for (auto j = 0; j < drbm_replica.hSize; j++) {
				auto gradient = this->dataMeanXH(drbm_replica, data, label, i, j, mujk) - drbm_replica.expectedValueXH(i, j, z, mujk);
				this->gradient.weightXH(i, j) += gradient * inv_batch_size;
			}
		}
		for (auto j = 0; j < drbm_replica.hSize; j++) {
			auto gradient = this->dataMeanH(drbm_replica, data, label, j, mujk) - drbm_replica.expectedValueH(j, z, mujk);
			this->gradient.biasH(j) += gradient * inv_batch_size;
		}
		for (auto j = 0; j < drbm_replica.hSize; j++) {
			for (auto k = 0; k < drbm_replica.ySize; k++) {
				auto gradient = this->dataMeanHY(drbm_replica, data, label, j, k, mujk) - drbm_replica.expectedValueHY(j, k, z, mujk);
				this->gradient.weightHY(j, k) += gradient * inv_batch_size;
			}
		}
		for (auto k = 0; k < drbm_replica.ySize; k++) {
			auto gradient = this->dataMeanY(drbm_replica, data, label, k, mujk) - drbm_replica.expectedValueY(k, z, mujk);
			this->gradient.biasY(k) += gradient * inv_batch_size;
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

	// update optimizer
	this->optimizer.iteration++;

}

double DRBM01Trainer::dataMeanXH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = data(xindex) * drbm.sigmoid(mu);
	//return value;
	return 0;
}

double DRBM01Trainer::dataMeanXH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	auto value = data(xindex) * drbm.sigmoid(mujk(hindex, label));
	return value;
}

double DRBM01Trainer::dataMeanH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = drbm.sigmoid(mu);
	//return value;
	return 0;
}

double DRBM01Trainer::dataMeanH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	auto value = drbm.sigmoid(mujk(hindex, label));
	return value;
}

double DRBM01Trainer::dataMeanHY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex)
{
	//if (yindex != label) return 0.0;
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = drbm.sigmoid(mu);
	//return value;
	return 0;
}

double DRBM01Trainer::dataMeanHY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & mujk)
{
	if (yindex != label) return 0.0;

	auto value = drbm.sigmoid(mujk(hindex, label));
	return value;
}

double DRBM01Trainer::dataMeanY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int yindex)
{
	//auto value = (yindex != label) ? 0.0 : 1.0;
	//return value;
	return 0;
}

double DRBM01Trainer::dataMeanY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}
