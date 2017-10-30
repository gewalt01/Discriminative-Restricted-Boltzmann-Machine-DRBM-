// 23test.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"


int main()
{
	std::vector<std::vector<double>> origin_data(8);
	origin_data[0] = std::vector<double>({ 0, 0, 0 });
	origin_data[1] = std::vector<double>({ 0, 0, 1 });
	origin_data[2] = std::vector<double>({ 0, 1, 0 });
	origin_data[3] = std::vector<double>({ 0, 1, 1 });
	origin_data[4] = std::vector<double>({ 1, 0, 0 });
	origin_data[5] = std::vector<double>({ 1, 0, 1 });
	origin_data[6] = std::vector<double>({ 1, 1, 0 });
	origin_data[7] = std::vector<double>({ 1, 1, 1 });

	int dup_num = 300;
	DRBM drbm(3 * dup_num, 10, 8);
	DRBMTrainer trainer(drbm);
	trainer.optimizer.alpha *= 1;

	int dataset_size = 8 * 1000;
	std::vector<Eigen::VectorXd> dataset(dataset_size);
	std::vector<Eigen::VectorXd> dataset_test(dataset_size);
	for (int i = 0; i < dataset.size(); i++) {
		dataset[i] = Eigen::VectorXd(3 * dup_num);
		dataset_test[i] = Eigen::VectorXd(3 * dup_num);
		for (int j = 0; j < dup_num; j++) {
			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());

			// 0.0以上1.0未満の値を等確率で発生させる
			std::uniform_real_distribution<> dist(-1.5, 1.5);

			dataset[i](3 * j + 0) = (origin_data[i % 8][0] + dist(engine)) * 1;
			dataset[i](3 * j + 1) = (origin_data[i % 8][1] + dist(engine)) * 1;
			dataset[i](3 * j + 2) = (origin_data[i % 8][2] + dist(engine)) * 1;

			dataset_test[i](3 * j + 0) = (origin_data[i % 8][0] + dist(engine)) * 1;
			dataset_test[i](3 * j + 1) = (origin_data[i % 8][1] + dist(engine)) * 1;
			dataset_test[i](3 * j + 2) = (origin_data[i % 8][2] + dist(engine)) * 1;
		}
	}

	std::vector<int> labelset(dataset_size);
	for (int i = 0; i < dataset_size; i++) {
		labelset[i] = i % 8;
	}



	for (int e = 0; e < 100000; e++) {
		std::vector<int> all_index(dataset.size());
		std::iota(all_index.begin(), all_index.end(), 0);
		std::random_device seed_gen;
		std::mt19937 engine(seed_gen());
		std::shuffle(all_index.begin(), all_index.end(), engine);

		int batch_size = 1;
		std::vector<int> batch_indexes(batch_size);
		std::copy(all_index.begin(), all_index.begin() + batch_size, batch_indexes.begin());

		trainer.train(drbm, dataset, labelset, batch_indexes);
		std::cout << e << ": ";
		std::uniform_int_distribution<> dist(0, dataset_size % 8);
		drbm.nodeX = dataset_test[dist(engine) + 0];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 1];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 2];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 3];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 4];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 5];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 6];
		std::cout << drbm.maxCondProbYIndex() << ", ";
		drbm.nodeX = dataset_test[dist(engine) + 7];
		std::cout << drbm.maxCondProbYIndex() << std::endl;
	}

    return 0;
}

