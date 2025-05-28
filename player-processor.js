class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.port.onmessage = (event) => {
      this.buffer.push(...event.data);
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    for (let i = 0; i < output.length; i++) {
      output[i] = this.buffer.length > 0 ? this.buffer.shift() : 0;
    }
    return true;
  }
}

registerProcessor("player-processor", PlayerProcessor);
