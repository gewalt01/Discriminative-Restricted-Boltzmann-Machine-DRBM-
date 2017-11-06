#pragma once
#include "Eigen/Core"
class SparseDRBM
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
	Eigen::VectorXd sparseH;
	Eigen::MatrixXd weightXH;
	Eigen::MatrixXd weightHY;

public:
	SparseDRBM();
	SparseDRBM(size_t xsize, size_t hsize, size_t ysize);
	~SparseDRBM();

	double normalizeConstant();
	double normalizeConstantDiv2H();

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
	double expectedValueAbsHExpSparse(int hindex);
	double expectedValueAbsHExpSparse(int hindex, double z);
	double expectedValueAbsHExpSparse(int hindex, double z, Eigen::MatrixXd & mujk);
};

