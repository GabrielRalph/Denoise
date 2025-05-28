
let session, audioContext, sourceNode, workletNode, stream;
let rnnState = null;
let rawChunks = [], denoisedChunks = [], inputBuffer = [];
let lastRaw = new Float32Array(480);
let lastDenoised = new Float32Array(480);
const attenLimDb = new Float32Array([0]);

const rawCanvas = document.getElementById("rawCanvas");
const denoisedCanvas = document.getElementById("denoisedCanvas");
const rawCtx = rawCanvas.getContext("2d");
const denoisedCtx = denoisedCanvas.getContext("2d");

function drawWaveform(ctx, data) {
  const w = ctx.canvas.width;
  const h = ctx.canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  for (let i = 0; i < data.length; i++) {
    const x = (i / data.length) * w;
    const y = h / 2 - data[i] * h / 2;
    ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function animate() {
  drawWaveform(rawCtx, lastRaw);
  drawWaveform(denoisedCtx, lastDenoised);
  requestAnimationFrame(animate);
}
animate();

function mergeChunks(buffers) {
  const length = buffers.reduce((sum, b) => sum + b.length, 0);
  const result = new Float32Array(length);
  let offset = 0;
  for (let b of buffers) {
    result.set(b, offset);
    offset += b.length;
  }
  return result;
}

function encodeMonoWAV(data, sampleRate) {
  const buffer = new ArrayBuffer(44 + data.length * 2);
  const view = new DataView(buffer);
  const writeString = (v, o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + data.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, data.length * 2, true);
  for (let i = 0; i < data.length; i++) {
    view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, data[i])) * 0x7FFF, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

function triggerDownload(blob, filename) {
  const a = document.createElement("a");
  const url = URL.createObjectURL(blob);
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function loadModel() {
  session = await ort.InferenceSession.create("https://cdn.jsdelivr.net/gh/G-R-Li/Denoise/denoiser_model.onnx");
  console.log("Model Loaded");
}

document.getElementById("start").onclick = async () => {
  await loadModel();
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext({ sampleRate: 48000 });
  await audioContext.audioWorklet.addModule("denoise-processor.js");

  sourceNode = audioContext.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioContext, "denoise-processor", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
  });

  workletNode.port.onmessage = async (event) => {
    const input = new Float32Array(event.data);
    inputBuffer.push(...input);

    while (inputBuffer.length >= 480) {
      const frame = inputBuffer.slice(0, 480);
      inputBuffer = inputBuffer.slice(480);
      lastRaw = Float32Array.from(frame);
      rawChunks.push(lastRaw);

      const inputTensor = new ort.Tensor("float32", lastRaw, [480]);
      if (!rnnState) {
        const dummy = await session.run({
          input_frame: inputTensor,
          states: new ort.Tensor("float32", new Float32Array(45304), [45304]),
          atten_lim_db: new ort.Tensor("float32", attenLimDb, [1]),
        });
        rnnState = new Float32Array(dummy["new_states"].data.length);
      }

      const feeds = {
        input_frame: inputTensor,
        states: new ort.Tensor("float32", rnnState, [rnnState.length]),
        atten_lim_db: new ort.Tensor("float32", attenLimDb, [1]),
      };

      try {
        const result = await session.run(feeds);
        rnnState = result["new_states"].data;
        lastDenoised = new Float32Array(result["enhanced_audio_frame"].data);
        denoisedChunks.push(lastDenoised);
      } catch (err) {
        console.error("Error", err);
      }
    }
  };

  sourceNode.connect(workletNode).connect(audioContext.destination);
  document.getElementById("start").disabled = true;
  document.getElementById("stop").disabled = false;
};

document.getElementById("stop").onclick = () => {
  stream.getTracks().forEach(track => track.stop());
  sourceNode.disconnect();
  workletNode.disconnect();
  audioContext.close();
  rnnState = null;

  const raw = mergeChunks(rawChunks);
  const clean = mergeChunks(denoisedChunks);
  triggerDownload(encodeMonoWAV(raw, 48000), "raw_audio.wav");
  setTimeout(() => {
    triggerDownload(encodeMonoWAV(clean, 48000), "denoised_audio.wav");
  }, 500);

  rawChunks = [];
  denoisedChunks = [];
  inputBuffer = [];
  document.getElementById("start").disabled = false;
  document.getElementById("stop").disabled = true;
  console.log("Ended, Audio Exported");
};
