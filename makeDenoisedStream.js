/* global ort */
const MODEL_PATH = "https://cdn.jsdelivr.net/gh/G-R-Li/Denoise/denoiser_model.onnx";
const STATE_SIZE = 45304;

export async function makeDenoisedStream(inputStream) {

  console.log("ORT Object:", ort);
  const audioContext = new AudioContext({ sampleRate: 48000 });
  await audioContext.audioWorklet.addModule("denoise-processor.js");
  await audioContext.audioWorklet.addModule("player-processor.js");

  const session = await ort.InferenceSession.create(MODEL_PATH);

  const attenLimDb = new Float32Array([0]);
  let rnnState = null;
  const inputAudioTrack = inputStream.getAudioTracks()[0];
  if (!inputAudioTrack) throw new Error("No audio track found in input stream");

  const source = audioContext.createMediaStreamSource(new MediaStream([inputAudioTrack]));
  const denoiseNode = new AudioWorkletNode(audioContext, "denoise-processor");
  const playerNode = new AudioWorkletNode(audioContext, "player-processor");
  const outputDest = audioContext.createMediaStreamDestination();

  source.connect(denoiseNode);
  playerNode.connect(outputDest);

  const inputBuffer = [];
  denoiseNode.port.onmessage = async (event) => {
    inputBuffer.push(...event.data);
    while (inputBuffer.length >= 480) {
      const frame = inputBuffer.splice(0, 480);
      const inputTensor = new ort.Tensor("float32", frame, [480]);

      if (!rnnState) {
        const result = await session.run({
          input_frame: inputTensor,
          states: new ort.Tensor("float32", new Float32Array(STATE_SIZE), [STATE_SIZE]),
          atten_lim_db: new ort.Tensor("float32", attenLimDb, [1]),
        });
        rnnState = result["new_states"].data;
      }

      try {
        const result = await session.run({
          input_frame: inputTensor,
          states: new ort.Tensor("float32", rnnState, [STATE_SIZE]),
          atten_lim_db: new ort.Tensor("float32", attenLimDb, [1]),
        });
        rnnState = result["new_states"].data;
        const enhanced = new Float32Array(result["enhanced_audio_frame"].data);
        playerNode.port.postMessage(enhanced);
      } catch (err) {
        console.error("Inference error:", err);
      }
    }
  };

  const outputAudioTrack = outputDest.stream.getAudioTracks()[0];
  outputAudioTrack.addEventListener("ended", () => {
    inputAudioTrack.stop();
  });

  const videoTracks = inputStream.getVideoTracks();
  return new MediaStream([outputAudioTrack, ...videoTracks]);
}
