let socket;
let mediaRecorder;
let audioChunks = [];

const connectBtn = document.getElementById('connectBtn');
const talkBtn = document.getElementById('talkBtn');
const status = document.getElementById('status');
const myLang = document.getElementById('myLang');

// 1. Connect and perform Handshake
connectBtn.onclick = () => {
  const userId = Math.random().toString(36).substring(7); // Random ID for testing
  socket = new WebSocket(`ws://localhost:8000/ws/call/testroom/${userId}`);
  socket.binaryType = "arraybuffer";

  socket.onopen = () => {
    status.innerText = "Status: Connected. Sending config...";
    // Sending the "Handshake" config
    socket.send(JSON.stringify({ native_lang: myLang.value }));
    connectBtn.disabled = true;
    talkBtn.disabled = false;
    status.innerText = "Status: Ready (User: " + userId + ")";
  };

  socket.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
      status.innerText = "Status: Receiving translation...";
      playAudio(event.data);
    } else {
      const msg = JSON.parse(event.data);
      console.log("Server message:", msg);
    }
  };
};

// 2. Audio Playback Logic
function playAudio(arrayBuffer) {
  const blob = new Blob([arrayBuffer], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
  audio.onended = () => status.innerText = "Status: Ready";
}

// 3. Recording Logic (Push-to-Talk style for cleaner demo)
talkBtn.onmousedown = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  // Whisper.cpp loves 16kHz. Most browsers record higher, but the server handles conversion.
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
  mediaRecorder.onstop = () => {
    const blob = new Blob(audioChunks, { type: 'audio/wav' });
    blob.arrayBuffer().then(buffer => {
      socket.send(buffer);
      status.innerText = "Status: Processing translation...";
    });
  };

  mediaRecorder.start();
  talkBtn.classList.add('speaking');
  status.innerText = "Status: Listening...";
};

talkBtn.onmouseup = () => {
  mediaRecorder.stop();
  talkBtn.classList.remove('speaking');
};