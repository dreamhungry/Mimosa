/**
 * AudioWorklet processor for capturing microphone audio.
 * This runs in a separate audio thread and sends PCM data to the main thread.
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._bufferSize = 4096;
        this._buffer = new Float32Array(this._bufferSize);
        this._writeIndex = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const channelData = input[0];

        for (let i = 0; i < channelData.length; i++) {
            this._buffer[this._writeIndex++] = channelData[i];

            if (this._writeIndex >= this._bufferSize) {
                // Send accumulated buffer to main thread
                this.port.postMessage({
                    type: 'audio',
                    buffer: this._buffer.slice(),
                });
                this._writeIndex = 0;
            }
        }

        return true;
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
