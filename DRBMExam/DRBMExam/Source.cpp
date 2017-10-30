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
	std::vector<Eigen::VectorXd> dataset_test(mnist_test.dataset.size());
	std::cout << drbm.normalizeConstantDiv2H() << std::endl;

	for (int n = 0; n < dataset.size(); n++) {
		set_data(dataset[n], mnist.dataset[n]);
	}

	for (int n = 0; n < dataset_test.size(); n++) {
		set_data(dataset_test[n], mnist_test.dataset[n]);
	}
	drbm.nodeX = dataset_test[0];
	auto z = drbm.normalizeConstantDiv2H();


	auto drbm_backup = drbm;
	auto drbm_delta = drbm;

	int epoch = 5000000;
	int batch_size = 5;
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

		drbm_delta = drbm;
		if (n % 1000 == 0) {
			double seikai = 0.0;
			std::vector<int> histgram(10);
			for (int n = 0; n < mnist_test.labelset.size(); n++) {
				drbm.nodeX = dataset_test[n];
				auto label = mnist_test.labelset[n];
				auto inference = drbm.maxCondProbYIndex();
				if (label == inference) seikai += 1.0;
				histgram[inference]++;
				//		std::cout << "inference: " << inference << ", label: " << label << std::endl;
			}
			for (int i = 0; i < histgram.size(); i++) {
				std::cout << "histgram[" << i << "]: " << histgram[i] << std::endl;
			}

			std::cout << "rate: " << seikai / mnist.labelset.size() << std::endl;
			std::cout << n << ": " << drbm.normalizeConstantDiv2H() << std::endl;
		}
	}

	double seikai = 0.0;
	std::vector<int> histgram(10);
	for (int n = 0; n < mnist_test.labelset.size(); n++) {
		drbm.nodeX = dataset_test[n];
		auto label = mnist_test.labelset[n];
		auto inference = drbm.maxCondProbYIndex();
		if (label == inference) seikai += 1.0;
		histgram[inference]++;
//		std::cout << "inference: " << inference << ", label: " << label << std::endl;
	}
	for (int i = 0; i < histgram.size(); i++) {
		std::cout << "histgram[" << i << "]: " << histgram[i] << std::endl;
	}

	std::cout << "rate: " << seikai / mnist_test.labelset.size() << std::endl;


	getchar();
	return 0;
}