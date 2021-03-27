#include <iostream> 
#include <fstream> 
#include "NeuralNetwork.h"
#include "math.h"
#include <omp.h>

NeuralNetwork::NeuralNetwork() {
	mConfusion = nullptr;
}

NeuralNetwork::~NeuralNetwork() {
	if (mConfusion)
		delete mConfusion;
}

NeuralNetwork::NeuralNetwork(
	vector<int> architecture,
	double learningRate, //Taxa de aprendizado
	Activation activation) { // Por padrão é utilizada a tangente hiperbólica
	init(architecture, learningRate, activation);
}

void NeuralNetwork::init(
	vector<int> architecture,
	double learningRate,
	Activation activation) {
	mArchitecture = architecture;
	mLearningRate = learningRate;
	mActivation = activation;

	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (unsigned int i = 0; i < architecture.size(); i++) {
			// Adiciona um neurônio extra a cada camada como viés (com peso = 1)
			int size = architecture[i] + (i != architecture.size() - 1);
			mNeurons.push_back(new RowVector(size));
			mErrors.push_back(new RowVector(size));

			if (i < architecture.size() - 1)
				// Define o multiplicador do viés como 1
				mNeurons.back()->coeffRef(architecture[i]) = 1.0;


			// Inicializa os pesos
			if (i > 0) {
				mWeights.push_back(new Matrix(architecture[i - 1] + 1, size));
				mWeights.back()->setRandom();
			}
		}
		mConfusion = new Matrix(architecture.back(), architecture.back());
}

double NeuralNetwork::activation(double x) {
	if (mActivation == TANH)
		return tanh(x);
	if (mActivation == SIGMOID)
		return 1.0 / (1.0 + exp(-x));
	return 0;
}

double NeuralNetwork::activationDerivative(double x) {
	if (mActivation == TANH)
		return 1 - tanh(x) * tanh(x);
	if (mActivation == SIGMOID)
		return x * (1.0 - x);
	return 0;
}

void NeuralNetwork::forward(RowVector& input) {

	// Define a entrada da primeira camada
	mNeurons.front()->block(0, 0, 1, input.size()) = input;

	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)

		// Propaga adiante (multiplicação de vetor)
		for (unsigned int i = 1; i < mArchitecture.size(); i++) {

			// Copia os valores ignorando o último neurônio porque ele é de viés
			mNeurons[i]->block(0, 0, 1, mArchitecture[i]) = (*mNeurons[i - 1] * *mWeights[i - 1]).block(0, 0, 1, mArchitecture[i]);

			// Aplica a função de ativação
			for (int col = 0; col < mArchitecture[i]; col++)
				mNeurons[i]->coeffRef(col) = activation(mNeurons[i]->coeffRef(col));
		}
}

void NeuralNetwork::test(RowVector& input, RowVector& output) {
	forward(input);

	// Calcula os erros da última camada
	*mErrors.back() = output - *mNeurons.back();
}

void NeuralNetwork::resetConfusion() {
	if (mConfusion)
		mConfusion->setZero();
}

void NeuralNetwork::evaluate(RowVector& output) {
	double desired = 0, actual = 0;
	mConfusion->coeffRef(
		vote(output, desired),
		vote(*mNeurons.back(), actual)
	)++;
}

void NeuralNetwork::confusionMatrix(RowVector*& precision, RowVector*& recall) {
	int rows = (int)mConfusion->rows();
	int cols = (int)mConfusion->cols();
		
	precision = new RowVector(cols);

	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int col = 0; col < cols; col++) {
			double colSum = 0;
			for (int row = 0; row < rows; row++)
				colSum += mConfusion->coeffRef(row, col);
			precision->coeffRef(col) = mConfusion->coeffRef(col, col) / colSum;
		}
	
	recall = new RowVector(rows);

	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int row = 0; row < rows; row++) {
			double rowSum = 0;
			for (int col = 0; col < cols; col++)
				rowSum += mConfusion->coeffRef(row, col);
			recall->coeffRef(row) = mConfusion->coeffRef(row, row) / rowSum;
		}
	

	// Converte a confusão (da matrix de confusão) para porcentagem
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int row = 0; row < rows; row++) {
			double rowSum = 0;
			for (int col = 0; col < cols; col++)
				rowSum += mConfusion->coeffRef(row, col);
			for (int col = 0; col < cols; col++)
				mConfusion->coeffRef(row, col) = mConfusion->coeffRef(row, col) * 100 / rowSum;
		}
}


void NeuralNetwork::backward(RowVector& output) {

	// Calcula os erros da última camada
	*mErrors.back() = output - *mNeurons.back();

	// Calcula os erros das camadas ocultas (multiplicação de vetores)
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (size_t i = mErrors.size() - 2; i > 0; i--)
			*mErrors[i] = *mErrors[i + 1] * mWeights[i]->transpose();


	// Atualiza os pesos
	size_t size = mWeights.size();
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (size_t i = 0; i < size; i++)
			for (int col = 0, cols = (int)mWeights[i]->cols(); col < cols; col++)
				for (int row = 0; row < mWeights[i]->rows(); row++) {
					mWeights[i]->coeffRef(row, col) +=
						mLearningRate *
						mErrors[i + 1]->coeffRef(col) *
						activationDerivative(mNeurons[i + 1]->coeffRef(col)) *
						mNeurons[i]->coeffRef(row);
				}
}

void NeuralNetwork::train(RowVector& input, RowVector& output) {
	forward(input);
	backward(output);
}

// Erro quadrático médio
double NeuralNetwork::mse() {
	return sqrt((*mErrors.back()).dot((*mErrors.back())) / mErrors.back()->size());
}

int NeuralNetwork::vote(double& value) {
	auto it = mNeurons.back();
	return vote(*it, value);
}

int NeuralNetwork::vote(RowVector& v, double& value) {
	int index = 0;
	#pragma omp parallel for firstprivate(x) private(i)  num_threads(2)
		for (int i = 1; i < v.cols(); i++)
			if (v[i] > v[index])
				index = i;
	value = v[index];
	return index;
}

double NeuralNetwork::output(int col) {
	auto it = mNeurons.back();
	return (*it)[col];
}

void NeuralNetwork::save(const char* filename) {
	stringstream tplgy;
	for (auto it = mArchitecture.begin(), _end = mArchitecture.end(); it != _end; it++)
		tplgy << *it << (it != _end - 1 ? "," : "");

	stringstream wts;
	for (auto it = mWeights.begin(), _end = mWeights.end(); it != _end; it++)
		wts << **it << (it != _end - 1 ? "," : "") << endl;

	ofstream file(filename);
	file << "learningRate: " << mLearningRate << endl;
	file << "architecture: " << tplgy.str() << endl;
	file << "activation: " << mActivation << endl;
	file << "weights: " << endl << wts.str() << endl;
	file.close();
}

bool NeuralNetwork::load(const char* filename) {
	mArchitecture.clear();

	ifstream file(filename);
	if (!file.is_open())
		return false;
	string line, name, value;
	if (!getline(file, line, '\n'))
		return false;
	stringstream lr(line);

	// Lê a taxa de aprendizado
	getline(lr, name, ':');
	if (name != "learningRate")
		return false;
	if (!getline(lr, value, '\n'))
		return false;
	mLearningRate = atof(value.c_str());

	// Lê a topologia
	getline(file, line, '\n');
	stringstream ss(line);
	getline(ss, name, ':');
	if (name != "architecture")
		return false;
	while (getline(ss, value, ','))
		mArchitecture.push_back(atoi(value.c_str()));

	// Lê a ativação
	getline(file, line, '\n');
	stringstream sss(line);
	getline(sss, name, ':');
	if (name != "activation")
		return false;
	if (!getline(sss, value, '\n'))
		return false;
	mActivation = (Activation)atoi(value.c_str());

	// Inicializa usando a arquitetura lida
	init(mArchitecture, mLearningRate, mActivation);

	// Lê os pesos
	getline(file, line, '\n');
	stringstream we(line);
	getline(we, name, ':');
	if (!(name == "weights"))
		return false;

	string matrix;
	for (int i = 0; i < mArchitecture.size(); i++)
		if (getline(file, matrix, ',')) {
			stringstream ss(matrix);
			int row = 0;
			while (getline(ss, value, '\n'))
				if (!value.empty()) {
					stringstream word(value);
					int col = 0;
					while (getline(word, value, ' '))
						if (!value.empty())
							mWeights[i]->coeffRef(row, col++) = atof(value.c_str());
					row++;
				}
		}

	file.close();
	return true;
}