#pragma once
#include "Eigen/Core"

class GeneralizedDRBM;

class GeneralizedDRBMOptimizer
{
	struct moment {
		Eigen::VectorXd biasH;
		Eigen::VectorXd biasY;
		Eigen::MatrixXd weightXH;
		Eigen::MatrixXd weightHY;
	};
public:
	double alpha = 0.001;
	double beta1 = 0.9;
	double beta2 = 0.999;
	double epsilon = 1E-08;
	int iteration = 1;

	struct moment moment1;
	struct moment moment2;

public:
	GeneralizedDRBMOptimizer();
	GeneralizedDRBMOptimizer(GeneralizedDRBM & drbm);
	~GeneralizedDRBMOptimizer();

	double deltaBiasH(int hindex, double gradient);
	double deltaBiasY(int yindex, double gradient);
	double deltaWeightXH(int xindex, int hindex, double gradient);
	double deltaWeightHY(int hindex, int yindex, double gradient);
};

