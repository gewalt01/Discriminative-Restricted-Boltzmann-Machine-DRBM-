#pragma once
#include "Eigen/Core"
#include "DRBMOptimizer.h"
class DRBM;

class DRBMTrainer
{
public:
	struct {
		Eigen::VectorXf biasC;
		Eigen::VectorXf biasD;
		Eigen::MatrixXf weightXH;
		Eigen::MatrixXf weightHY;
	} gradient;

	DRBMOptimizer optimizer;

public:
	DRBMTrainer();
	DRBMTrainer(DRBM & drbm);
	~DRBMTrainer();

	void train(DRBM & drbm, Eigen::VectorXf & data, int label);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXf & data, int label, int xindex, int hindex);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXf & data, int label, int xindex, int hindex, Eigen::MatrixXf & muJK);
	double dataMeanH(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex);
	double dataMeanH(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, Eigen::MatrixXf & muJK);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, int yindex);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXf & data, int label, int hindex, int yindex, Eigen::MatrixXf & muJK);
	double dataMeanY(DRBM & drbm, Eigen::VectorXf & data, int label, int yindex);
	double dataMeanY(DRBM & drbm, Eigen::VectorXf & data, int label, int yindex, Eigen::MatrixXf & muJK);
};

