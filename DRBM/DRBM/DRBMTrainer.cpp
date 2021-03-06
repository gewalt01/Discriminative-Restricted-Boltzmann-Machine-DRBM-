﻿#include "DRBMTrainer.h"
#include "DRBM.h"
#include <omp.h>

DRBMTrainer::DRBMTrainer()
{
}

DRBMTrainer::DRBMTrainer(DRBM & drbm)
{
	this->gradient.biasH.setConstant(drbm.hSize, 0.0);
	this->gradient.biasY.setConstant(drbm.ySize, 0.0);
	this->gradient.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->gradient.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->optimizer = DRBMOptimizer(drbm);
}


DRBMTrainer::~DRBMTrainer()
{
}

void DRBMTrainer::train(DRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes)
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
		auto z = drbm_replica.normalizeConstantDiv2H(mujk);

		// Gradient
		for (auto i = 0; i < drbm_replica.xSize; i++) {
			for (auto j = 0; j < drbm_replica.hSize; j++) {
				auto gradient = this->dataMeanXH(drbm_replica, data, label, i, j, mujk) - drbm_replica.expectedValueXH(i, j, z, mujk);
				this->gradient.weightXH(i, j) += gradient;
			}
		}
		for (auto j = 0; j < drbm_replica.hSize; j++) {
			auto gradient = this->dataMeanH(drbm_replica, data, label, j, mujk) - drbm_replica.expectedValueH(j, z, mujk);
			this->gradient.biasH(j) += gradient;
		}
		for (auto j = 0; j < drbm_replica.hSize; j++) {
			for (auto k = 0; k < drbm_replica.ySize; k++) {
				auto gradient = this->dataMeanHY(drbm_replica, data, label, j, k, mujk) - drbm_replica.expectedValueHY(j, k, z, mujk);
				this->gradient.weightHY(j, k) += gradient;
			}
		}
		for (auto k = 0; k < drbm_replica.ySize; k++) {
			auto gradient = this->dataMeanY(drbm_replica, data, label, k, mujk) - drbm_replica.expectedValueY(k, z, mujk);
			this->gradient.biasY(k) += gradient;
		}
	}

	//  * inv_batch_size
	this->gradient.biasH *= inv_batch_size;
	this->gradient.biasY *= inv_batch_size;
	this->gradient.weightXH *= inv_batch_size;
	this->gradient.weightHY *= inv_batch_size;

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

double DRBMTrainer::dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = data(xindex) * tanh(mu);
	//return value;
	return 0;
}

double DRBMTrainer::dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	auto value = data(xindex) * tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex)
{
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = tanh(mu);
	//return value;
	return 0;
}

double DRBMTrainer::dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	auto value = tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex)
{
	//if (yindex != label) return 0.0;
	//// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	//auto value = tanh(mu);
	//return value;
	return 0;
}

double DRBMTrainer::dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & mujk)
{
	if (yindex != label) return 0.0;

	auto value = tanh(mujk(hindex, label));
	return value;
}

double DRBMTrainer::dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex)
{
	//auto value = (yindex != label) ? 0.0 : 1.0;
	//return value;
	return 0;
}

double DRBMTrainer::dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}
