#define EIGEN_NO_DEBUG // コード内のassertを無効化．
#define EIGEN_DONT_PARALLELIZE // 並列を無効化．

#include "DRBM.h"
#include "DRBMOptimizer.h"
#include "DRBMTrainer.h"
#include "DRBM01.h"
#include "DRBM01Optimizer.h"
#include "DRBM01Trainer.h"
#include "mnist.h"
#include <algorithm>
#include <numeric>
#include <random>
#include "Eigen/Core"
#include <fstream>
#include <iostream>
#include <vector>
#include "Eigen/Core"

struct option {
	int drbmType = 0;
	int batchSize = 1;
	int hiddenSize = 1;
	double alpha = 1.0;
	int eachCrossVaridation = 500;
};

struct option config_option_from_cin();

template<class RBM, class TRAINER>
void run(RBM & drbm, TRAINER & trainer, struct option & opt);

int main(void) {
	auto opt = config_option_from_cin();

	switch (opt.drbmType) {
	case 0:  // DRBM{-1, +1}
		{
			DRBM drbm(784, opt.hiddenSize, 10);
			DRBMTrainer trainer(drbm);
			run(drbm, trainer, opt);
		}
		break;
	case 1:  // DRBM{0. 1}
		{
			DRBM01 drbm(784, opt.hiddenSize, 10);
			DRBM01Trainer trainer(drbm);
			run(drbm, trainer, opt);
		}
		break;
	}

	return 0;
}

option config_option_from_cin()
{
	struct option opt;
	std::cout << "DRBM{-1, +1} -> 0" << std::endl;
	std::cout << "DRBM{0, 1} -> 1" << std::endl;
	std::cout << "DRBM:";
	std::cin >> opt.drbmType;

	std::cout << "hidden_size:";
	std::cin >> opt.hiddenSize;

	std::cout << "batch_size:";
	std::cin >> opt.batchSize;

	std::cout << "alpha(learning_rate):";
	std::cin >> opt.alpha;

	return opt;
}

template <class RBM, class TRAINER>
void run(RBM & drbm, TRAINER & trainer, struct option & opt) {
	trainer.optimizer.alpha *= opt.alpha;
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
	std::cout << drbm.normalizeConstant() << std::endl;

	for (int n = 0; n < dataset.size(); n++) {
		set_data(dataset[n], mnist.dataset[n]);
	}

	for (int n = 0; n < dataset_test.size(); n++) {
		set_data(dataset_test[n], mnist_test.dataset[n]);
	}
	drbm.nodeX = dataset_test[0];
	auto z = drbm.normalizeConstant();


	auto drbm_backup = drbm;
	auto drbm_delta = drbm;

	auto test = [](auto & drbm, auto & dataset, auto & labelset) {
		double seikai = 0.0;
		std::vector<int> histgram(10);
#pragma omp parallel for
		for (int n = 0; n < labelset.size(); n++) {
			auto drbm_replica = drbm;
			drbm_replica.nodeX = dataset[n];
			auto label = labelset[n];
			auto inference = drbm_replica.maxCondProbYIndex();
			if (label == inference) seikai += 1.0;
			histgram[inference]++;
			//		std::cout << "inference: " << inference << ", label: " << label << std::endl;
		}
		for (int i = 0; i < histgram.size(); i++) {
			std::cout << "histgram[" << i << "]: " << histgram[i] << std::endl;
		}

		double rate = seikai / labelset.size();

		return rate;
	};

	int epoch = 5000000;
	double max_rate_traindata = 0.0;
	double max_rate_testdata = 0.0;
	std::ofstream rate_traindata_file("rate_traindata.txt", std::ios::out | std::ios::trunc);
	std::ofstream rate_testdata_file("rate_testdata.txt", std::ios::out | std::ios::trunc);

	for (int n = 1; n < epoch; n++) {
		std::vector<int> all_index(dataset.size());
		std::iota(all_index.begin(), all_index.end(), 0);
		std::random_device seed_gen;
		std::mt19937 engine(seed_gen());
		std::shuffle(all_index.begin(), all_index.end(), engine);

		std::vector<int> batch_indexes(opt.batchSize);
		std::copy(all_index.begin(), all_index.begin() + opt.batchSize, batch_indexes.begin());

		trainer.train(drbm, dataset, mnist.labelset, batch_indexes);
		//std::cout << n << ": " << drbm.normalizeConstantDiv2H() << std::endl;
		std::cout << n << std::endl;

		drbm_delta = drbm;
		if (n % opt.eachCrossVaridation == 0) {
			double rate;
			//rate = test(drbm, dataset, mnist.labelset);
			//max_rate_traindata = std::max(rate, max_rate_traindata);
			//std::cout << "rate(train): " << rate << std::endl;
			//std::cout << "max_rate(train): " << max_rate_traindata << std::endl;
			//rate_traindata_file << rate << std::endl;

			rate = test(drbm, dataset_test, mnist_test.labelset);
			max_rate_testdata = std::max(rate, max_rate_testdata);
			std::cout << "rate(test): " << rate << std::endl;
			std::cout << "max_rate(test): " << max_rate_testdata << std::endl;
			rate_testdata_file << rate << std::endl;

			std::cout << n << ": " << drbm.normalizeConstant() << std::endl;
		}
	}

	double rate = test(drbm, dataset_test, mnist_test.labelset);
	std::cout << "rate: " << rate << std::endl;
}