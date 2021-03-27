# Trabalho Prático da disciplina de Programação Paralela e Concorrente

## Thiago Luigi Gonçalves Lima <br>

<div style="text-align: justify">

### <b>Descrição</b>

A área de inteligência artificial (IA) está sendo bastante utilizada para geração de sistemas inteligentes, mas envolve o conhecimento de várias outras áreas e possui muitos algoritmos envolvidos. É necessário o conhecimento em estatística, computação, negócio e matemática. Exemplos de algoritmos utilizados são:

1. Kmeans <br>
2. KNN <br>
3. Random Forest <br>
4. Support Vector Machines - SVM <br>
5. Apriori <br>
6. Redes Neurais Artificiais <br>
7. Algoritmos genéticos <br>
8. Naive Bayes <br>
9. PageRank <br>

Na literatura existem várias versões sequenciais e paralelas desses algoritmos que podem ser consultadas.

Encontre um problema onde um desses algoritmos possa ser aplicado. 

Em seguida, implemente uma versão paralela de um dos algoritmos acima usando uma ou mais bibliotecas de paralelização estudadas na disciplina e a linguagem de programação de sua preferência.
O trabalho deverá ser submetido como um arquivo compactado, contendo o código fonte comentado e um vídeo de no máximo 5 minutos explicando como o problema foi resolvido. O arquivo deverá conter o nome e um sobrenome do aluno ("NomeSobrenome.zip").
Trabalhos  que não compilam receberão nota ZERO, bem como trabalhos que sejam considerados como plágio. Deve ser adicionado no zip um arquivo README para explicar qual problema está sendo resolvido, qual o algoritmo escolhido e  como funciona a compilação e a execução do programa implementado.

Exemplos de ferramentas para gravar vídeo: extensões do Chrome (Loom, Screencastify), CAMTASIA STUDIO, Movie Maker, OBS, etc.

A avaliação será realizada  da seguinte forma: 60% para código implementado e 40% para vídeo de apresentação.

O trabalho deverá ser realizado individualmente.

### <b>O Trabalho</b>

Para o trabalho foi escolhido o algoritmo de Rede Neural Artificial desenvolvido por [Hatem Mostafa](https://www.codeproject.com/Members/Hatem-Mostafa). Este algoritmo é um simples contador que irá receber um número em binário como entrada e deverá retornar essa entrada + 1. O algoritmo supracitado foi implementado de forma sequencial e para o trabalho irei paralelizá-lo utilizando o OpenMP. <br><br>

Abaixo tem uma tabela mostrando um exemplo de como deve ser o comportamento do algoritmo.

<table>
    <tr>
        <th>Entrada</th>
        <th>Saída</th>
    </tr>
    <tr>
        <td> 0 0 0 </td>
        <td> 0 0 1 </td>
    </tr>
    <tr>
        <td> 0 0 1 </td>
        <td> 0 1 0 </td>
    </tr>
    <tr>
        <td> 0 1 0 </td>
        <td> 0 1 1 </td>
    </tr>
    <tr>
        <td> 0 1 1 </td>
        <td> 1 0 0 </td>
    </tr>
    <tr>
        <td> 1 0 0 </td>
        <td> 1 0 1 </td>
    </tr>
    <tr>
        <td> 1 0 1 </td>
        <td> 1 1 0 </td>
    </tr>
    <tr>
        <td> 1 1 0 </td>
        <td> 1 1 1 </td>
    </tr>
    <tr>
        <td> 1 1 1 </td>
        <td> 0 0 0 </td>
    </tr>

</table>

Mostafa quando implementou o algoritmo utilizou uma biblioteca chamada [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page). E por causa disso é necessário incluir uma flag no momento da compilação. A biblioteca já está presente no trabalho na pasta ``eigen-3.3.9``. Abaixo segue um exemplo de como compilar o programa caso você já esteja na pasta raíz do projeto: <br><br>

<div style="text-align: left">

``g++ -I eigen-3.3.9/ CounterNeuralNetwork/main.cpp CounterNeuralNetwork/NeuralNetwork.cpp -o programa`` <br><br>

</div>

A parte ``-I eigen-3.3.9`` diz respeito a biblioteca que precisa ser incluída no processo de compilação. Para executar basta digitar no terminal ``./programa`` caso você esteja na pasta raíz do projeto. <br>
O algoritmo original parece ter sido implementado para um ambiente Windows (visual c++). Mas para este trabalho estou utilizando o ``Ubuntu 20.04``, o ``g++ 9.3.0`` e o ``OpenMP``. Para compilar e executar utilizando o OpenMP (ou seja, a versão modificada deste trabalho) basta utilizar o seguinte comando: <br><br>

<div style="text-align: left">

``mpic++ -I eigen-3.3.9/ CounterNeuralNetwork/main.cpp CounterNeuralNetwork/NeuralNetwork.cpp -o programa`` <br><br>

E para executar: <br><br>

``mpirun -np 2 programa`` <br><br>

</div>

No comando acima 2 significa a quantidade de processos que serão utilizados. Eu optei por 2 mas fique a vontade pra escolher outro valor. <br>

No Ubuntu, caso você não tenha as ferramentas necessárias, para instalar basta utilizar o seguinte comando: <br><br>

<div style="text-align: left">

``sudo apt install build-essential libopenmpi-dev`` <br><br>

</div>

A execução não é rápida uma vez que é executada quase 50 mil vezes para treinamento. <br><br>

### <b>Referências</b>

Eu recomendo muito que acessem a página onde Mostafa disponibilizou o código. Além de estar tudo organizado e ser possível pegar o código fonte na íntegra (eu removi alguns arquivos relacionados ao ambiente Windows) é possível ver outros exemplos e ilustrações que podem ajudar a entender não só o código aqui trabalhado mas também o próprio conceito de redes neurais artificiais. A página está em inglês. <br><br>

[Artificial Neural Network C++ Class - CodeProject](https://www.codeproject.com/Articles/5292985/Artificial-Neural-Network-Cplusplus-class#SimpleCounter)

</div>