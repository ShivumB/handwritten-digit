public void drawNet(NeuralNet net) {
  background(255);

  noStroke();
  fill(0, 255, 255, 150);
  rect(0, 0, width, height);

  for (int i = 0; i < net.nodes.length; i++) {

    float ySpacing = (float)(height)/net.nodes[i].length;

    for (int j = 0; j < net.nodes[i].length; j++) {
      
      noStroke();
      fill(125 + 125*net.nodes[i][j].value);
      ellipse(50 + i*100, ySpacing*j, ySpacing/2, ySpacing/2);


      if (i > 1) {
        for (int k = 0; k < net.nodes[i-1].length; k++) {
          
          float prevYSpacing = (float)(height)/net.nodes[i-1].length;
          
          if(net.nodes[i][j].weights[k] > 0) {
            stroke(0,0,0);
          } else {
            stroke(255,0,0);
          }
          strokeWeight(pow(abs(net.nodes[i][j].weights[k]),3));
          line(50 + i*100, ySpacing*j, 50 + (i-1)*100, prevYSpacing*k);
          
          
        }
      }
    }
  }
};
