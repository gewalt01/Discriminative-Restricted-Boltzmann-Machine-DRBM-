#include "GeneralizedSparseDRBMOptimizer.h"
#include "GeneralizedSparseDRBM.h"


GeneralizedSparseDRBMOptimizer::GeneralizedSparseDRBMOptimizer()
{
}

GeneralizedSparseDRBMOptimizer::GeneralizedSparseDRBMOptimizer(GeneralizedSparseDRBM & drbm)
{
	this->moment1.biasH.setConstant(drbm.hSize, 0.0);
	this->moment1.biasY.setConstant(drbm.ySize, 0.0);
	this->moment1.sparseH.setConstant(drbm.hSize, 0.0);
	this->moment1.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->moment1.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);

	this->moment2.biasH.setConstant(drbm.hSize, 0.0);
	this->moment2.biasY.setConstant(drbm.ySize, 0.0);
	this->moment2.sparseH.setConstant(drbm.hSize, 0.0);
	this->moment2.weightXH.setConstant(drbm.xSize, drbm.hSize, 0.0);
	this->moment2.weightHY.setConstant(drbm.hSize, drbm.ySize, 0.0);
}


GeneralizedSparseDRBMOptimizer::~GeneralizedSparseDRBMOptimizer()
{
}

double GeneralizedSparseDRBMOptimizer::deltaBiasH(int hindex, double gradient)
{
	// Adamax
	auto m = this->moment1.biasH(hindex) = this->beta1 * this->moment1.biasH(hindex) + (1.0 - this->beta1) * gradient;
	auto v = this->moment2.biasH(hindex) = std::max(this->beta2 * this->moment2.biasH(hindex), abs(gradient));
	auto delta = this->alpha / (1.0 - pow(this->beta1, this->iteration)) * m / (v + this->epsilon);

	return delta;
}

double GeneralizedSparseDRBMOptimizer::deltaBiasY(int yindex, double gradient)
{
	// Adamax
	auto m = this->moment1.biasY(yindex) = this->beta1 * this->moment1.biasY(yindex) + (1.0 - this->beta1) * gradient;
	auto v = this->moment2.biasY(yindex) = std::max(this->beta2 * this->moment2.biasY(yindex), abs(gradient));
	auto delta = this->alpha / (1.0 - pow(this->beta1, this->iteration)) * m / (v + this->epsilon);

	return delta;
}

double GeneralizedSparseDRBMOptimizer::deltaSparseH(int hindex, double gradient)
{
	// Adamax
	auto m = this->moment1.sparseH(hindex) = this->beta1 * this->moment1.sparseH(hindex) + (1.0 - this->beta1) * gradient;
	auto v = this->moment2.sparseH(hindex) = std::max(this->beta2 * this->moment2.sparseH(hindex), abs(gradient));
	auto delta = this->alpha / (1.0 - pow(this->beta1, this->iteration)) * m / (v + this->epsilon);

	return delta;
}

double GeneralizedSparseDRBMOptimizer::deltaWeightXH(int xindex, int hindex, double gradient)
{
	// Adamax
	auto m = this->moment1.weightXH(xindex, hindex) = this->beta1 * this->moment1.weightXH(xindex, hindex) + (1.0 - this->beta1) * gradient;
	auto v = this->moment2.weightXH(xindex, hindex) = std::max(this->beta2 * this->moment2.weightXH(xindex, hindex), abs(gradient));
	auto delta = this->alpha / (1.0 - pow(this->beta1, this->iteration)) * m / (v + this->epsilon);

	return delta;
}

double GeneralizedSparseDRBMOptimizer::deltaWeightHY(int hindex, int yindex, double gradient)
{
	// Adamax
	auto m = this->moment1.weightHY(hindex, yindex) = this->beta1 * this->moment1.weightHY(hindex, yindex) + (1.0 - this->beta1) * gradient;
	auto v = this->moment2.weightHY(hindex, yindex) = std::max(this->beta2 * this->moment2.weightHY(hindex, yindex), abs(gradient));
	auto delta = this->alpha / (1.0 - pow(this->beta1, this->iteration)) * m / (v + this->epsilon);

	return delta;
}
