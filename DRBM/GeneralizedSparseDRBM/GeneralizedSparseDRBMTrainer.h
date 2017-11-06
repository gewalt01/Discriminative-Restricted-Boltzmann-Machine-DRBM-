#pragma once
#include "Eigen/Core"
#include "GeneralizedSparseDRBMOptimizer.h"
#include <vector>
class GeneralizedSparseDRBM;

class GeneralizedSparseDRBMTrainer
{
public:
	struct {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::VectorXd sparseH;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	GeneralizedSparseDRBMOptimizer optimizer;

public:
	GeneralizedSparseDRBMTrainer();
	GeneralizedSparseDRBMTrainer(GeneralizedSparseDRBM & drbm);
	~GeneralizedSparseDRBMTrainer();

	void train(GeneralizedSparseDRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes);
	double dataMeanXH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & mujk);
	double dataMeanH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk);
	double dataMeanHY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & mujk);
	double dataMeanY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & mujk);
	double dataMeanAbsHExpSparse(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanAbsHExpSparse(GeneralizedSparseDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & mujk);
};

