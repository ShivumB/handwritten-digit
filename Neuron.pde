public class Neuron {

  public float bias = random(-1, 1);
  public float pBias = 0;

  public float value = 0;
  public float pValue = 0;

  public float[] weights;
  public float[] pWeights;

  Neuron(int wtNum) {
    weights = new float[wtNum];
    pWeights = new float[wtNum];

    for (int i = 0; i < weights.length; i++) {
      weights[i] = random(-1, 1); 
      pWeights[i] = 0;
    }
  }

  Neuron(float value, float[] weights) {
    this.value = value;
    this.weights = weights;
  }

  public void activate() {
    value = 1/(1 + exp(-value));
  }
}
