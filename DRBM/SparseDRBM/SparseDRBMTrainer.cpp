#include "SparseDRBMTrainer.h"
#include "SparseDRBM.h"

SparseDRBMTrainer::SparseDRBMTrainer()
{
}

SparseDRBMTrainer::SparseDRBMTrainer(SparseDRBM & drbm)
{
	this->gradient.biasH.setConstant(drbm.hSize, 0.0);
	this->gradient.biasY.setConstant(drbm.ySize, 0.0);
	this->gradient.sparseH.setConstant(drbm.hSize, 0.0);
	this->gradient.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->gradient.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->optimizer = SparseDRBMOptimizer(drbm);
}


SparseDRBMTrainer::~SparseDRBMTrainer()
{
}

void SparseDRBMTrainer::train(SparseDRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes)
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

		auto z = drbm.normalizeConstantDiv2H();

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

double SparseDRBMTrainer::dataMeanXH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex)
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

double SparseDRBMTrainer::dataMeanXH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	//auto mu = drbm.biasH(hindex) + drbm.weightHY(hindex, label);
	//for (auto i = 0; i < drbm.xSize; i++) {
	//	mu += drbm.weightXH(i, hindex) * data[i];
	//}
	auto value = data(xindex) * tanh(mujk(hindex, label));
	return value;
}

double SparseDRBMTrainer::dataMeanH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex)
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

double SparseDRBMTrainer::dataMeanH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk)
{
	// FIXME: muの計算使いまわしできそうだけど…
	auto value = tanh(mujk(hindex, label));
	return value;
}

double SparseDRBMTrainer::dataMeanHY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex)
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

double SparseDRBMTrainer::dataMeanHY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & mujk)
{
	if (yindex != label) return 0.0;

	auto value = tanh(mujk(hindex, label));
	return value;
}

double SparseDRBMTrainer::dataMeanY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}

double SparseDRBMTrainer::dataMeanY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK)
{
	auto value = (yindex != label) ? 0.0 : 1.0;
	return value;
}

double SparseDRBMTrainer::dataMeanAbsHExpSparse(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex)
{
	auto value = -exp(drbm.sparseH(hindex));
	return value;
}

double SparseDRBMTrainer::dataMeanAbsHExpSparse(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK)
{
	auto value = -exp(drbm.sparseH(hindex));
	return value;
}
