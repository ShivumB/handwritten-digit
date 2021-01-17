public class NeuralNet {

  Neuron nodes[][] = new Neuron[3][];
  float step = 0.05;

  NeuralNet() {

    nodes[0] = new Neuron[784];

    nodes[1] = new Neuron[16];

    nodes[2] = new Neuron[10];

    for (int i = 0; i < nodes.length; i++) {
      for (int j = 0; j < nodes[i].length; j++) {
        if (i == 0) {
          nodes[i][j] = new Neuron(0);
        } else {
          nodes[i][j] = new Neuron(nodes[i-1].length);
        }
      }
    }
  }

  public float cost(float[] a, float[] b) {
    float c = 0;
    int ans = 0;
  
    //find the right answer
    for(int i = 1; i < b.length; i++) {
      if(b[i] > ans) {
        ans = i;
      }
    }
    
    c = sq(a[ans] - b[ans]) * 9/18;

    for (int i = 0; i < a.length; i++) {
        c += sq(a[i]-b[i]) * 1 / 9;
    }

    return c;
  }

  public float[] run(Neuron[][] nodes) {

    for (int i = 1; i < nodes.length; i++) {

      for (int j = 0; j < nodes[i].length; j++) {

        nodes[i][j].value = 0;

        for (int k = 0; k < nodes[i-1].length; k++) {
          nodes[i][j].value += nodes[i-1][k].value*nodes[i][j].weights[k];
        }

        nodes[i][j].value += nodes[i][j].bias;

        nodes[i][j].activate();
      }
    }

    float outputLayer[] = new float[nodes[nodes.length-1].length];

    for (int i = 0; i < outputLayer.length; i++) {
      outputLayer[i] = nodes[nodes.length-1][i].value;
    }

    return outputLayer;
  }

  public float[] run(float[] input) {

    for (int i = 0; i < input.length; i++) {
      nodes[0][i].value = input[i];
    }

    return run(nodes);
  }

  public void learn(float[]input, float[] output) {    
    float cost = cost(run(input), output);

    //del(cost) / del(last neuron layer)
    for (int i = 0; i < nodes[nodes.length-1].length; i++) {
      nodes[nodes.length-1][i].value += step;      

      //last neuron layer as an array
      float[] neuronArr = new float[nodes[nodes.length-1].length];

      for (int j = 0; j < neuronArr.length; j++) {
        neuronArr[j] = nodes[nodes.length-1][j].value;
      }

      float newCost = cost(neuronArr, output);

      nodes[nodes.length-1][i].pValue = (newCost - cost)/step;

      nodes[nodes.length-1][i].value -= step;
    }

    //starting from the output layer, go left; differentiate the output layer's weights, then its biases, then the nodes of the previous layer until the previous layer is the input layer
    for (int i = nodes.length - 1; i > 0; i--) {

      //each node
      for (int j = 0; j < nodes[i].length; j++) {

        //each weight/neuron in the further left layer
        for (int k = 0; k < nodes[i][j].weights.length; k++) {

          //differentiate the weights first
          //change the input
          nodes[i][j].weights[k] += step;

          //see the effect on this node
          float newValue = 0;
          for (int ef = 0; ef < nodes[i][j].weights.length; ef++) {
            newValue += nodes[i-1][ef].value*nodes[i][j].weights[ef];
          }
          newValue += nodes[i][j].bias;
          newValue = 1/(1 + exp(-newValue));

          //find the derivative
          nodes[i][j].pWeights[k] += (newValue - nodes[i][j].value)/step * nodes[i][j].pValue;

          //fix the input
          nodes[i][j].weights[k] -= step;

          //now, differentiate each of the neurons in the further left layer
          //change the input
          nodes[i-1][k].value += step;

          //see the result
          newValue = 0;
          for (int ef = 0; ef < nodes[i][j].weights.length; ef++) {
            newValue += nodes[i-1][ef].value*nodes[i][j].weights[ef];
          }
          newValue += nodes[i][j].bias;
          newValue = 1/(1 + exp(-newValue));

          //find the derivative
          nodes[i-1][k].pValue = (newValue - nodes[i][j].value)/step * nodes[i][j].pValue;

          //fix the input
          nodes[i-1][k].value -= step;
        }

        //this node's bias
        nodes[i][j].bias += step;

        //see the effect
        float newValue = 0;
        for (int ef = 0; ef < nodes[i][j].weights.length; ef++) {
          newValue += nodes[i-1][ef].value*nodes[i][j].weights[ef];
        }
        newValue += nodes[i][j].bias;
        newValue = 1/(1 + exp(-newValue));

        //find the derivative
        nodes[i][j].pBias += (newValue - nodes[i][j].value)/step * nodes[i][j].pValue;

        //fix the input
        nodes[i][j].bias -= step;
      }
    }
  }

  public void update() {
    //apply all the changes - this part should be changed to maximize efficiency while preserving integrity
    //currently, it learns from one example, updates, and proceeds; it should learn from a random number of samples before updating

    //go left to right, update biases then links
    for (int i = 0; i < nodes.length; i++) {

      //each node
      for (int j = 0; j < nodes[i].length; j++) {

        nodes[i][j].bias -= nodes[i][j].pBias*step;
        nodes[i][j].pBias = 0;

        //each weight
        for (int k = 0; k < nodes[i][j].weights.length; k++) {
          nodes[i][j].weights[k] -= nodes[i][j].pWeights[k]*step;
          nodes[i][j].pWeights[k] = 0;
        }
      }
    }
    
    
    
  }
}
