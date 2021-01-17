NeuralNet net = new NeuralNet();

Table train[] = new Table[6];
Table test;

PrintWriter writeNet;

public void train(int exampleNum) {

  writeNet = createWriter("net.txt");


  println("Reading training data.");

  for (int i = 0; i < exampleNum; i++) {
    train[i] = loadTable("mnist_train-" + (i+1) + ".csv", "csv");
  }

  println("Learning.");

  for (int i = 0; i < exampleNum; i++) {
    for (int j = 0; j < 10000; j++) {

      float output[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
      output[train[i].getInt(j, 0)] = 1;

      float input[] = new float[784];
      for (int k = 1; k < 785; k++) {
        input[k - 1] = train[i].getInt(j, k)/255;
      }

      net.learn(input, output);

      if (j % 500 == 0) {
        println("table " + (i+1) + ", line " + j);
      }

      if (j % 500 == 0) {
        println(net.nodes[2][0].value + ", " + net.nodes[2][0].pValue);
        net.update();
      }
    }
  }

  println("Recording neural net."); 

  for (int i = 1; i < net.nodes.length; i++) {
    for (int j = 0; j < net.nodes[i].length; j++) {
      writeNet.print(net.nodes[i][j].bias);

      for (int k = 0; k < net.nodes[i][j].weights.length; k++) {
        writeNet.print(",");
        writeNet.print(net.nodes[i][j].weights[k]);
      }

      writeNet.print("n");
    }

    writeNet.print("l");
  }

  writeNet.flush();
  writeNet.close();

  println("Finished.");
}

public void load() {
  String[] fileData = loadStrings("net.txt");

  String[] layers = split(fileData[0], "l");

  String[][] neurons = new String[layers.length][];

  for (int i = 0; i < layers.length; i++) {
    neurons[i] = split(layers[i], "n");
  }

  String[][][] actualData = new String[layers.length][][];

  for (int i = 0; i < layers.length; i++) {    

    actualData[i] = new String[neurons[i].length][];

    for (int j = 0; j < neurons[i].length; j++) {

      String[] temp = split(neurons[i][j], ",");

      actualData[i][j] = temp;
    }
  }

  for (int i = 1; i < net.nodes.length; i ++) {
    for (int j = 0; j < net.nodes[i].length; j++) {

      net.nodes[i][j].bias = parseFloat(actualData[i-1][j][0]);

      for (int k = 1; k < actualData[i-1][j].length; k++) {
        net.nodes[i][j].weights[k-1] = parseFloat(actualData[i-1][j][k]);
      }
    }
  }
};

public void test() {


  println("Reading testing data.");

  test = loadTable("mnist_test.csv", "csv");
  int wins = 0;
  float avgCost = 0;
  float avgConf = 0;
  float avgHighConf = 0;

  for (int i = 0; i < 10000; i++) {

    int ans = test.getInt(i, 0);

    float input[] = new float[784];
    for (int j = 1; j < 785; j++) {
      input[j - 1] = test.getInt(i, j)/255;
    }

    int guess = 0;
    float[] output = net.run(input);

    float[] rightOutput = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    rightOutput[ans] = 1;
    avgCost += net.cost(output, rightOutput)/10000;

    for (int j = 0; j < output.length-1; j++) {

      avgConf += output[j]/10;

      if (output[j] > output[j+1]) {
        guess = j;
      } else {
        guess = j + 1;
      }
    }

    avgHighConf += output[guess];

    if (guess == ans) {
      wins ++;
    }
  }

  println("Accuracy: " + wins/10000f);
  println("Average cost: " + avgCost);
  println("Average confidence: " + avgConf/10000);
  println("Average highest confidence: " + avgHighConf/10000);
};

void setup() {

  size(600, 600);
  background(255);

  //train(1);

  load();

  test();
}
