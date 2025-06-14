import { relURL } from "../usefull-funcs.js";

let script = document.createElement("script");
script.src = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";
document.head.appendChild(script);

// import ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/+esm'
/* global ort */
const MODEL_PATH = relURL("denoiser_model.onnx", import.meta);
const STATE_SIZE = 45304;


export async function createDenoisedTrack(inputAudioTrack) {
  if (!inputAudioTrack) throw new Error("No audio track provided.");

  if (!window.ort) {
    await new Promise((resolve) => {
      script.onload = resolve;
      script.onerror = () => {
        throw new Error("Failed to load ONNX Runtime Web script.");
      };
    });
    console.log("ORT Object:", ort);
  }

  
  const audioContext = new AudioContext({ sampleRate: 48000 });

  // Load the audio worklet modules
  await Promise.all([
    audioContext.audioWorklet.addModule(relURL("denoise-processor.js", import.meta)),
    audioContext.audioWorklet.addModule(relURL("player-processor.js", import.meta))
  ]);

  // Create the AudioWorkletNodes
  const denoiseNode = new AudioWorkletNode(audioContext, "denoise-processor");
  const playerNode = new AudioWorkletNode(audioContext, "player-processor");

  // Create a MediaStreamSource from the input audio track
  const source = audioContext.createMediaStreamSource(new MediaStream([inputAudioTrack]));
  source.connect(denoiseNode); // Connect the source to the denoise node

  // Create a MediaStreamDestination to output the processed audio
  const outputDest = audioContext.createMediaStreamDestination();
  playerNode.connect(outputDest); // Connect the player node to the output destination

  // Setup onnxruntime session
  const session = await ort.InferenceSession.create(MODEL_PATH);
  const attenLimDb = new Float32Array([0]);
  let rnnState = null;

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

  inputAudioTrack.addEventListener("ended", () => {
    outputAudioTrack.stop();
  });
  return outputAudioTrack;
}

export async function makeDenoisedStream(inputStream) {
  if (!inputStream || !inputStream.getAudioTracks().length) {
    throw new Error("Input stream must contain an audio track.");
  }

  let inputAudioTrack = inputStream.getAudioTracks()[0];
  let denoisedTrack = await createDenoisedTrack(inputAudioTrack);

  // replace the original audio track with the denoised one
  inputStream.removeTrack(inputAudioTrack);
  inputStream.addTrack(denoisedTrack);
}
