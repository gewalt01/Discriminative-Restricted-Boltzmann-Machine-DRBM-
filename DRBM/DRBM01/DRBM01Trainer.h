#pragma once
#include "Eigen/Core"
#include "DRBM01Optimizer.h"
#include <vector>
class DRBM01;

class DRBM01Trainer
{
public:
	struct {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	DRBM01Optimizer optimizer;

public:
	DRBM01Trainer();
	DRBM01Trainer(DRBM01 & drbm);
	~DRBM01Trainer();

	void train(DRBM01 & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes);
	double dataMeanXH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanHY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(DRBM01 & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK);
};

