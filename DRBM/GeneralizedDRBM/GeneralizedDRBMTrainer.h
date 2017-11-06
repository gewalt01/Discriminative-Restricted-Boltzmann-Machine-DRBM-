#pragma once
#include "Eigen/Core"
#include "GeneralizedDRBMOptimizer.h"
#include <vector>
class GeneralizedDRBM;

class GeneralizedDRBMTrainer
{
public:
	struct {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	} gradient;

	GeneralizedDRBMOptimizer optimizer;

public:
	GeneralizedDRBMTrainer();
	GeneralizedDRBMTrainer(GeneralizedDRBM & drbm);
	~GeneralizedDRBMTrainer();

	void train(GeneralizedDRBM & drbm, std::vector<Eigen::VectorXd> & dataset, std::vector<int> & labelset, std::vector<int> & batch_indexes);
	double dataMeanXH(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex);
	double dataMeanXH(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int xindex, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanH(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int hindex);
	double dataMeanH(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, Eigen::MatrixXd & muJK);
	double dataMeanHY(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex);
	double dataMeanHY(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int hindex, int yindex, Eigen::MatrixXd & muJK);
	double dataMeanY(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int yindex);
	double dataMeanY(GeneralizedDRBM & drbm, Eigen::VectorXd & data, int label, int yindex, Eigen::MatrixXd & muJK);
};

