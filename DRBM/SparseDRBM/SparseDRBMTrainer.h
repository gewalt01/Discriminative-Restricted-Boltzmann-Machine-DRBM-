#pragma once
#include "Eigen/Core"
#include "SparseDRBMOptimizer.h"
#include <vector>
class SparseDRBM;

class SparseDRBMTrainer
{
public:
	struct {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::VectorXd sparseH;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	SparseDRBMOptimizer optimizer;

public:
	SparseDRBMTrainer();
	SparseDRBMTrainer(SparseDRBM & drbm);
	~SparseDRBMTrainer();

	void train(SparseDRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes);
	double dataMeanXH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanHY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanAbsHExpSparse(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanAbsHExpSparse(SparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
};

