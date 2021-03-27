// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "NeuralNetwork.h"
#include <omp.h>

double unary(double x) {
	return x > .8 ? 1 : x < .2 ? 0 : x;
}

void test(NeuralNetwork& net) {
	cout << "Testing:" << endl;

	RowVector input(3), output(3);
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int num = 0; num < 8; num++) {
			input.coeffRef(0) = (num >> 2) & 1;
			input.coeffRef(1) = (num >> 1) & 1;
			input.coeffRef(2) = num & 1;

			output.coeffRef(0) = ((num + 1) >> 2) & 1;
			output.coeffRef(1) = ((num + 1) >> 1) & 1;
			output.coeffRef(2) = (num + 1) & 1;

			net.test(input, output);

			double mse = net.mse();
			cout << "In [" << input << "] "
				<< " Desired [" << output << "] "
				<< " Out [" << net.mNeurons.back()->unaryExpr(ptr_fun(unary)) << "] "
				<< " MSE [" << mse << "]" << endl;
		}
}

void train(NeuralNetwork& net) {
	cout << "Training:" << endl;
	RowVector input(3), output(3);
	int stop = 0;
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int i = 0; stop < 8 && i < 50000; i++) {
			cout << i + 1 << endl;
			for (int num = 0; stop < 8 && num < 8; num++) {
				input.coeffRef(0) = (num >> 2) & 1;
				input.coeffRef(1) = (num >> 1) & 1;
				input.coeffRef(2) = num & 1;

				output.coeffRef(0) = ((num + 1) >> 2) & 1;
				output.coeffRef(1) = ((num + 1) >> 1) & 1;
				output.coeffRef(2) = (num + 1) & 1;

				net.train(input, output);
				double mse = net.mse();
				cout << "In [" << input << "] "
					<< " Desired [" << output << "] "
					<< " Out [" << net.mNeurons.back()->unaryExpr(ptr_fun(unary)) << "] "
					<< " MSE [" << mse << "]" << endl;
				stop = mse < 0.1 ? stop + 1 : 0;
			}
		}
}

int main() {
	NeuralNetwork net({ 3, 5, 3 }, 0.05, NeuralNetwork::Activation::TANH);
	
	train(net);
	test(net);
	net.save("params.txt");

	cout << endl << "Neurons:" << endl;
	for(int i = 0; i < net.mNeurons.size(); i++)
		cout << *net.mNeurons[i] << endl;
	cout << endl << "Weights:" << endl;
	for (int i = 0; i < net.mWeights.size(); i++)
		cout << *net.mWeights[i] << endl;

	return 0;
}
