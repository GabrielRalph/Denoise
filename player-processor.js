class PlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = [];
    this.port.onmessage = (event) => {
      this.buffer.push(...event.data);
    };
  }

  process(inputs, outputs) {
    const channels = outputs[0];
    const numValues = Math.max(...channels.map(channel => channel.length));

    // console.log(`Processing ${numValues} values for ${channels.length} channels.`);
    let sum = 0;
    for (let i = 0; i < numValues; i++) {
      let value = this.buffer.length > 0 ? this.buffer.shift() : 0;
      sum += value**2;
      for (let channel = 0; channel < channels.length; channel++) {
        channels[channel][i] = value;
      }
    }

    // console.log(`Sum of processed values: ${new Array(Math.round(sum * 2)).fill("-").join("")}`);
    return true;
  }
}

registerProcessor("player-processor", PlayerProcessor);
