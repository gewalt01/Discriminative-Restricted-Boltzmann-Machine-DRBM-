#pragma once
#include "Eigen/Core"
#include <vector>
class GeneralizedDRBM
{
public:
	size_t  xSize;
	size_t  hSize;
	size_t  ySize;
	double hMin = 0.0;
	double hMax = 1.0;
	size_t  divSize = 1; // 分割数
	std::vector <double> hiddenValueSet;  // 隠れ変数の取りうる値
	Eigen::VectorXd nodeX;
	Eigen::VectorXd nodeH;
	Eigen::VectorXd nodeY;
	Eigen::VectorXd biasH;
	Eigen::VectorXd biasY;
	Eigen::MatrixXd weightXH;
	Eigen::MatrixXd weightHY;

public:
	GeneralizedDRBM();
	GeneralizedDRBM(size_t xsize, size_t hsize, size_t ysize);
	~GeneralizedDRBM();

	double normalizeConstant();

	double muJ(int hindex);
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

	// exp(mu)の可視変数に関する全ての実現値の総和
	double miniNormalizeConstantHidden(int hindex);
	double miniNormalizeConstantHidden(int hindex, int yindex);

	// 隠れ変数の取りうる値を返す
	std::vector<double> splitHiddenSet();

	// 隠れ変数の取りうるパターン数
	int getHiddenValueSetSize();

	// 隠れ変数の取りうる最大値を取得
	double getHiddenMax();

	// 隠れ変数の取りうる最大値を設定
	void setHiddenMax(double value);

	// 隠れ変数の取りうる最小値を取得
	double getHiddenMin();

	// 隠れ変数の取りうる最小値を設定
	void setHiddenMin(double value);

	// 隠れ変数の区間分割数を返す
	size_t getHiddenDivSize();

	// 隠れ変数の区間分割数を設定
	void setHiddenDivSize(size_t div_size);

};

