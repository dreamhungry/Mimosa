/**
 * WebSocket client for Mimosa.
 * Handles connection lifecycle, message sending/receiving, and auto-reconnection.
 */
class MimosaWebSocket {
    constructor(url) {
        this.url = url;
        this.ws = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;

        // Event callbacks
        this.onConnected = null;
        this.onDisconnected = null;
        this.onMessage = null;
        this.onError = null;
    }

    /**
     * Establish WebSocket connection.
     */
    connect() {
        try {
            this.ws = new WebSocket(this.url);

            this.ws.onopen = () => {
                console.log('[WS] Connected to server');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                if (this.onConnected) this.onConnected();
            };

            this.ws.onclose = (event) => {
                console.log('[WS] Disconnected:', event.code, event.reason);
                this.isConnected = false;
                if (this.onDisconnected) this.onDisconnected();
                this._attemptReconnect();
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    if (this.onMessage) this.onMessage(message);
                } catch (e) {
                    console.error('[WS] Failed to parse message:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.error('[WS] Error:', error);
                if (this.onError) this.onError(error);
            };
        } catch (e) {
            console.error('[WS] Connection failed:', e);
            this._attemptReconnect();
        }
    }

    /**
     * Send a JSON message.
     * @param {object} message - Message object to send.
     */
    send(message) {
        if (!this.isConnected || !this.ws) {
            console.warn('[WS] Not connected, cannot send message');
            return false;
        }

        try {
            const json = JSON.stringify(message);
            const sizeKB = (json.length / 1024).toFixed(1);
            if (message.type !== 'audio-data') {
                console.log(`[WS] Sending: type=${message.type}, size=${sizeKB}KB`);
            }
            this.ws.send(json);
            return true;
        } catch (e) {
            console.error(`[WS] Send failed (type=${message.type}, size~${(JSON.stringify(message).length / 1024).toFixed(0)}KB):`, e);
            return false;
        }
    }

    /**
     * Send text input message.
     * @param {string} text - User text input.
     */
    sendText(text) {
        return this.send({ type: 'text-input', text: text });
    }

    /**
     * Send audio data in chunks to avoid oversized WebSocket messages.
     * @param {Array<number>} audioData - Float32 audio samples.
     */
    sendAudioData(audioData) {
        const CHUNK_SIZE = 16000; // 1 second of audio per message
        const totalChunks = Math.ceil(audioData.length / CHUNK_SIZE);
        console.log(
            `[WS] Sending audio: ${audioData.length} samples in ${totalChunks} chunks`
        );

        for (let i = 0; i < audioData.length; i += CHUNK_SIZE) {
            const chunk = audioData.slice(i, i + CHUNK_SIZE);
            const success = this.send({ type: 'audio-data', audio: chunk });
            if (!success) {
                console.error(`[WS] Failed to send audio chunk ${Math.floor(i / CHUNK_SIZE) + 1}/${totalChunks}`);
                return false;
            }
        }
        console.log('[WS] All audio chunks sent successfully');
        return true;
    }

    /**
     * Signal end of audio recording.
     */
    sendAudioEnd() {
        return this.send({ type: 'audio-end' });
    }

    /**
     * Start real-time voice mode.
     */
    sendRealtimeStart() {
        console.log('[WS] Sending realtime-start');
        return this.send({ type: 'realtime-start' });
    }

    /**
     * Send real-time audio chunk.
     * @param {Array<number>} audioData - Float32 audio samples.
     */
    sendRealtimeAudio(audioData) {
        return this.send({ type: 'realtime-audio', audio: audioData });
    }

    /**
     * Stop real-time voice mode.
     */
    sendRealtimeStop() {
        console.log('[WS] Sending realtime-stop');
        return this.send({ type: 'realtime-stop' });
    }

    /**
     * Send interrupt signal.
     */
    sendInterrupt() {
        return this.send({ type: 'interrupt' });
    }

    /**
     * Disconnect from server.
     */
    disconnect() {
        this.maxReconnectAttempts = 0; // Prevent reconnection
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    /**
     * Attempt automatic reconnection with exponential backoff.
     */
    _attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('[WS] Max reconnect attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

        setTimeout(() => {
            this.connect();
        }, delay);
    }
}
