/**
 * Browser PCM WAV Audio Recorder
 * Captures microphone audio using the Web Audio API and encodes standard 16-bit PCM WAV.
 */

export class WavRecorder {
  constructor() {
    this.audioContext = null;
    this.mediaStream = null;
    this.sourceNode = null;
    this.processorNode = null;
    this.audioData = [];
    this.isRecording = false;
    this.sampleRate = 22050; // Standard ML sample rate
  }

  async start() {
    this.audioData = [];
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.sampleRate,
    });

    this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
    this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processorNode.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const channel = e.inputBuffer.getChannelData(0);
      this.audioData.push(new Float32Array(channel));
    };

    this.sourceNode.connect(this.processorNode);
    this.processorNode.connect(this.audioContext.destination);
    this.isRecording = true;
  }

  async stop() {
    this.isRecording = false;

    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode = null;
    }
    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = null;
    }
    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }

    return this.exportWav();
  }

  exportWav() {
    let totalLength = 0;
    for (const chunk of this.audioData) {
      totalLength += chunk.length;
    }

    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this.audioData) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    const wavBuffer = this.encodeWAV(merged, this.sampleRate);
    return new Blob([wavBuffer], { type: "audio/wav" });
  }

  encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    this.writeString(view, 0, "RIFF");
    view.setUint32(4, 36 + samples.length * 2, true);
    this.writeString(view, 8, "WAVE");

    // fmt sub-chunk
    this.writeString(view, 12, "fmt ");
    view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
    view.setUint16(20, 1, true); // AudioFormat (1 = PCM)
    view.setUint16(22, 1, true); // NumChannels (1 = Mono)
    view.setUint32(24, sampleRate, true); // SampleRate
    view.setUint32(28, sampleRate * 2, true); // ByteRate (SampleRate * 1 * 2)
    view.setUint16(32, 2, true); // BlockAlign (1 * 2)
    view.setUint16(34, 16, true); // BitsPerSample (16 bits)

    // data sub-chunk
    this.writeString(view, 36, "data");
    view.setUint32(40, samples.length * 2, true);

    // Write 16-bit PCM samples
    let index = 44;
    for (let i = 0; i < samples.length; i++) {
      let s = Math.max(-1, Math.min(1, samples[i]));
      s = s < 0 ? s * 0x8000 : s * 0x7fff;
      view.setInt16(index, s, true);
      index += 2;
    }

    return buffer;
  }

  writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }
}
