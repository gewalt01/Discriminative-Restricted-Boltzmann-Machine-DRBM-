#pragma once
#define EIGEN_NO_DEBUG // コード内のassertを無効化．
#define EIGEN_DONT_PARALLELIZE // 並列を無効化．#include "Eigen/Core"
#include "DRBMOptimizer.h"
#include <vector>
class DRBM;

class DRBMTrainer
{
public:
	struct {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	DRBMOptimizer optimizer;

public:
	DRBMTrainer();
	DRBMTrainer(DRBM & drbm);
	~DRBMTrainer();

	void train(DRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(DRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(DRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(DRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK);
};

