#include "DRBM.h"
#include "DRBMOptimizer.h"
#include "DRBMTrainer.h"
#include "mnist.h"
#include <algorithm>
#include <random>
#include "Eigen/Core"
#include <iostream>

int main(void) {
	DRBM drbm(784, 300, 10);
	DRBMTrainer trainer(drbm);
	trainer.optimizer.alpha *= 0.1;
	std::string train_data("train-images.idx3-ubyte"), train_label("train-labels.idx1-ubyte");
	std::string test_data("t10k-images.idx3-ubyte"), test_label("t10k-labels.idx1-ubyte");
	Mnist mnist(train_data, train_label);
	Mnist mnist_test(test_data, test_label);

	// シャッフル
	std::vector<int> indexes(mnist.labelset.size());
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());
	std::shuffle(indexes.begin(), indexes.end(), engine);
	auto set_data = [](auto & eigen, auto & data) {
		for (int i = 0; i < data.size(); i++) {
			eigen(i) = data[i];
		}
	};
	Eigen::VectorXf eigen(784);
	std::cout << drbm.normalizeConstantDiv2H() << std::endl;
	set_data(eigen, mnist.dataset[0]);
	drbm.nodeX = eigen;
	auto z = drbm.normalizeConstantDiv2H();

	//std::cout << "--------- <xh> ---------";
	//for (int i = 0; i < drbm.xSize; i++) {
	//	for (int j = 0; j < drbm.hSize; j++) {
	//		std::cout << drbm.expectedValueXH(i, j, z) << std::endl;
	//	}
	//}

	//std::cout << "--------- <h> ---------";
	//for (int j = 0; j < drbm.hSize; j++) {
	//	std::cout << drbm.expectedValueH(j, z) << std::endl;
	//}

	//std::cout << "--------- <hy> ---------";
	//for (int j = 0; j < drbm.hSize; j++) {
	//	for (int k = 0; k < drbm.ySize; k++) {
	//		std::cout << drbm.expectedValueHY(j, k, z) << std::endl;
	//	}
	//}


	//std::cout << "--------- <y> ---------";
	//for (int k = 0; k < drbm.ySize; k++) {
	//	std::cout << drbm.expectedValueY(k, z) << std::endl;
	//}


	//std::cout << "--------- MuJK ---------";
	//for (int j = 0; j < drbm.hSize; j++) {
	//	for (int k = 0; k < drbm.ySize; k++) {
	//		std::cout << drbm.muJK(j, k) << std::endl;
	//	}
	//}



	for (int n = 0; n < indexes.size(); n++) {
		set_data(eigen, mnist.dataset[n]);
		auto label = mnist.labelset[n];
		trainer.train(drbm, eigen, label);
		std::cout << n << ": " << drbm.normalizeConstantDiv2H() << std::endl;
	}



	getchar();
	return 0;
}