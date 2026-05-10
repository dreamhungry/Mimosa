/**
 * Audio recording and playback for Mimosa.
 * Handles microphone capture and audio response playback.
 */
class AudioManager {
    constructor() {
        this.mediaStream = null;
        this.audioContext = null;
        this.processor = null;
        this.isRecording = false;
        this.audioBuffer = [];

        // Playback
        this.audioQueue = [];
        this.isPlaying = false;

        // Callbacks
        this.onAudioData = null;
        this.onRecordingStart = null;
        this.onRecordingStop = null;
        this.onPlaybackEnd = null;
    }

    /**
     * Initialize audio context and request microphone permission.
     * Uses native sample rate to avoid browser compatibility issues,
     * then resamples to 16kHz before sending.
     */
    async init() {
        try {
            // Do NOT force sampleRate - many browsers ignore it and produce silence
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.nativeSampleRate = this.audioContext.sampleRate;
            this.targetSampleRate = 16000;
            console.log('[Audio] AudioContext initialized, native sample rate:', this.nativeSampleRate);
            return true;
        } catch (e) {
            console.error('[Audio] Failed to initialize AudioContext:', e);
            return false;
        }
    }

    /**
     * Downsample audio from native rate to target rate (16kHz).
     * @param {Float32Array} buffer - Input audio at native sample rate.
     * @returns {Float32Array} Resampled audio at target rate.
     */
    _resample(buffer) {
        if (this.nativeSampleRate === this.targetSampleRate) {
            return buffer;
        }
        const ratio = this.nativeSampleRate / this.targetSampleRate;
        const newLength = Math.round(buffer.length / ratio);
        const result = new Float32Array(newLength);
        for (let i = 0; i < newLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, buffer.length - 1);
            const frac = srcIndex - srcIndexFloor;
            result[i] = buffer[srcIndexFloor] * (1 - frac) + buffer[srcIndexCeil] * frac;
        }
        return result;
    }

    /**
     * Start recording from microphone.
     */
    async startRecording() {
        if (this.isRecording) return;

        try {
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });

            // Ensure audio context is running
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // Use ScriptProcessor for simplicity (AudioWorklet is better but more complex)
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            this.audioBuffer = [];

            this.processor.onaudioprocess = (event) => {
                if (!this.isRecording) return;
                const inputData = event.inputBuffer.getChannelData(0);
                // Resample from native rate to 16kHz
                const resampled = this._resample(inputData);
                const chunk = Array.from(resampled);
                this.audioBuffer.push(...chunk);

                if (this.onAudioData) {
                    this.onAudioData(chunk);
                }
            };

            source.connect(this.processor);
            this.processor.connect(this.audioContext.destination);

            this.isRecording = true;
            console.log(`[Audio] Recording started (native: ${this.nativeSampleRate}Hz -> target: ${this.targetSampleRate}Hz)`);
            if (this.onRecordingStart) this.onRecordingStart();

        } catch (e) {
            console.error('[Audio] Failed to start recording:', e);
            throw e;
        }
    }

    /**
     * Stop recording and return collected audio data.
     * @returns {Array<number>} Collected audio samples.
     */
    stopRecording() {
        if (!this.isRecording) return [];

        this.isRecording = false;

        // Clean up
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        const audioData = [...this.audioBuffer];
        this.audioBuffer = [];

        console.log(`[Audio] Recording stopped, ${audioData.length} samples collected`);
        if (this.onRecordingStop) this.onRecordingStop();

        return audioData;
    }

    /**
     * Play audio from base64 encoded data.
     * @param {string} base64Audio - Base64 encoded audio (MP3).
     */
    async playAudio(base64Audio) {
        if (!base64Audio) return;

        try {
            // Decode base64 to ArrayBuffer
            const binaryString = atob(base64Audio);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }

            // Ensure audio context is running
            if (this.audioContext.state === 'suspended') {
                await this.audioContext.resume();
            }

            // Decode and play
            const audioBuffer = await this.audioContext.decodeAudioData(bytes.buffer);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);

            source.onended = () => {
                this.isPlaying = false;
                if (this.onPlaybackEnd) this.onPlaybackEnd();
                this._playNext();
            };

            this.isPlaying = true;
            source.start(0);
            console.log('[Audio] Playing audio response');

        } catch (e) {
            console.error('[Audio] Playback failed:', e);
            this.isPlaying = false;
            this._playNext();
        }
    }

    /**
     * Queue audio for sequential playback.
     * @param {string} base64Audio - Base64 encoded audio.
     */
    queueAudio(base64Audio) {
        this.audioQueue.push(base64Audio);
        if (!this.isPlaying) {
            this._playNext();
        }
    }

    /**
     * Play next audio in queue.
     */
    _playNext() {
        if (this.audioQueue.length === 0) return;
        const next = this.audioQueue.shift();
        this.playAudio(next);
    }

    /**
     * Stop all playback and clear queue.
     */
    stopPlayback() {
        this.audioQueue = [];
        this.isPlaying = false;
    }
}
