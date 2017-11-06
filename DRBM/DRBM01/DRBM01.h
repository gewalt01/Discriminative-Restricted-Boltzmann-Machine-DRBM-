#pragma once
#include "Eigen/Core"
class DRBM01
{
public:
	size_t  xSize;
	size_t  hSize;
	size_t  ySize;
	Eigen::VectorXd nodeX;
	Eigen::VectorXd nodeH;
	Eigen::VectorXd nodeY;
	Eigen::VectorXd biasH;
	Eigen::VectorXd biasY;
	Eigen::MatrixXd weightXH;
	Eigen::MatrixXd weightHY;

public:
	DRBM01();
	DRBM01(size_t xsize, size_t hsize, size_t ysize);
	~DRBM01();

	double normalizeConstant();
	double normalizeConstant(Eigen::MatrixXd & mujk);

	double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

	double muJK(int hindex, int yindex);
	Eigen::MatrixXd muJKMatrix();

	double condProbY(int yindex);
	double condProbY(int yindex, double z);
	int maxCondProbYIndex();

	double expectedValueXH(int xindex, int hindex);
	double expectedValueXH(int xindex, int hindex, double z);
	double expectedValueXH(int xindex, int hindex, double z, Eigen::MatrixXd & mujk);
	double expectedValueH(int hindex);
	double expectedValueH(int hindex, double z);
	double expectedValueH(int hindex, double z, Eigen::MatrixXd & mujk);
	double expectedValueHY(int hindex, int yindex);
	double expectedValueHY(int hindex, int yindex, double z);
	double expectedValueHY(int hindex, int yindex, double z, Eigen::MatrixXd & mujk);
	double expectedValueY(int yindex);
	double expectedValueY(int yindex, double z);
	double expectedValueY(int yindex, double z, Eigen::MatrixXd & mujk);
};

