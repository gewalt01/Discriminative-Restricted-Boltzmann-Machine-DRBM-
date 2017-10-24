#include "DRBM.h"
#include "DRBMOptimizer.h"
#include "DRBMTrainer.h"
#include "mnist.h"
#include <algorithm>
#include <numeric>
#include <random>
#include "Eigen/Core"
#include <iostream>
#include <vector>
#include "Eigen/Core"

int main(void) {
	DRBM drbm(784, 50, 10);
	DRBMTrainer trainer(drbm);
	trainer.optimizer.alpha *= 1;
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
		eigen.resize(784);
		for (int i = 0; i < data.size(); i++) {
			eigen(i) = data[i];
		}
	};
	std::vector<Eigen::VectorXd> dataset(mnist.dataset.size());
	std::cout << drbm.normalizeConstantDiv2H() << std::endl;

	for (int n = 0; n < dataset.size(); n++) {
		set_data(dataset[n], mnist.dataset[n]);
	}
	drbm.nodeX = dataset[0];
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


	int epoch = 60000;
	int batch_size = 1;
	for (int n = 0; n < epoch; n++) {
		std::vector<int> all_index(dataset.size());
		std::iota(all_index.begin(), all_index.end(), 0);
		std::random_device seed_gen;
		std::mt19937 engine(seed_gen());
		std::shuffle(all_index.begin(), all_index.end(), engine);

		std::vector<int> batch_indexes(batch_size);
		std::copy(all_index.begin(), all_index.begin() + batch_size, batch_indexes.begin());

		trainer.train(drbm, dataset, mnist.labelset, batch_indexes);
		//std::cout << n << ": " << drbm.normalizeConstantDiv2H() << std::endl;
		std::cout << n << std::endl;
	}

	double seikai = 0.0;
	for (int n = 0; n < mnist_test.labelset.size(); n++) {
		drbm.nodeX = dataset[n];
		auto label = mnist_test.labelset[n];
		auto inference = drbm.maxCondProbYIndex();
		if (label == inference) seikai += 1.0;
		std::cout << "inference: " << inference << ", label: " << label << std::endl;
	}

	std::cout << "rate: " << seikai / mnist_test.labelset.size() << std::endl;


	getchar();
	return 0;
}