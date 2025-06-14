import { makeDenoisedStream } from "./makeDenoisedStream.js";


async function start() {
    console.log("Starting denoising...");
    
    let inputStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    
    if (!inputStream || !inputStream.getAudioTracks().length) {
        throw new Error("Input stream must contain an audio track.");
    }

    await makeDenoisedStream(inputStream)
    let chunks = [];
    const mediaRecorder = new MediaRecorder(inputStream);
    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            chunks.push(event.data);
        }
    };

    mediaRecorder.start();

    await new Promise((resolve) => setTimeout(resolve, 5000));

    await new Promise((resolve) => {
        mediaRecorder.onstop = resolve;
        mediaRecorder.stop();
    });

    const blob = new Blob(chunks, { type: "audio/ogg; codecs=opus" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play();
}

const button = document.getElementById("start")

button.addEventListener("click", async () => {
    button.disabled = true;
    try {
        await start();
    } catch (err) {
        console.error("Error during denoising:", err);
    } finally {
        button.disabled = false;
    }
});