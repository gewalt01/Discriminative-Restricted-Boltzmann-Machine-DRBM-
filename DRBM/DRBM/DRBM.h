#pragma once
#include "Eigen/Core"
class DRBM
{
public:
	size_t  xSize;
	size_t  hSize;
	size_t  ySize;
	Eigen::VectorXf nodeX;
	Eigen::VectorXf nodeH;
	Eigen::VectorXf nodeY;
	Eigen::VectorXf biasC;
	Eigen::VectorXf biasD;
	Eigen::MatrixXf weightXH;
	Eigen::MatrixXf weightHY;

public:
	DRBM();
	DRBM(size_t xsize, size_t hsize, size_t ysize);
	~DRBM();

	double normalizeConstant();
	double normalizeConstantDiv2H();

	double muJK(int hindex, int yindex);

	double condProbY(int yindex);
	double condProbY(int yindex, double z);

	double expectedValueXH(int xindex, int hindex);
	double expectedValueXH(int xindex, int hindex, double z);
	double expectedValueH(int hindex);
	double expectedValueH(int hindex, double z);
	double expectedValueHY(int hindex, int yindex);
	double expectedValueHY(int hindex, int yindex, double z);
	double expectedValueY(int yindex);
	double expectedValueY(int yindex, double z);
};

