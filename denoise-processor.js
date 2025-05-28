
class DenoiseProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0][0];
    if (!input) return true;
    const frame = Array.from(input);
    this.port.postMessage(frame);
    return true;
  }
}
registerProcessor("denoise-processor", DenoiseProcessor);
