#pragma once
#include "Eigen/Core"
#include "DRBMOptimizer.h"
class DRBM;

class DRBMTrainer
{
public:
	struct {
		Eigen::VectorXd biasC;
		Eigen::VectorXd biasD;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	DRBMOptimizer optimizer;

public:
	DRBMTrainer();
	DRBMTrainer(DRBM & drbm);
	~DRBMTrainer();

	void train(DRBM & drbm, Eigen::VectorXd & data, int label);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK);
};

