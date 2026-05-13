/**
 * Mimosa main application.
 * Coordinates WebSocket, Audio, Live2D, and UI interactions.
 */
console.log('[App] app.js loaded');
(function () {
    'use strict';

    // Determine WebSocket URL from current page location
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;

    // Managers
    const ws = new MimosaWebSocket(wsUrl);
    const audio = new AudioManager();
    const live2d = new Live2DManager('live2dCanvas');
    const interaction = new InteractionManager(live2d, ws);

    // DOM elements
    const connectionDot = document.getElementById('connectionDot');
    const statusText = document.getElementById('statusText');
    const emotionBadge = document.getElementById('emotionBadge');
    const chatMessages = document.getElementById('chatMessages');
    const textInput = document.getElementById('textInput');
    const btnSend = document.getElementById('btnSend');
    const btnMic = document.getElementById('btnMic');

    /**
     * Initialize the application.
     */
    async function init() {
        // Initialize audio
        await audio.init();

        // Initialize Live2D (model will be loaded after WebSocket connection)
        await live2d.init(null);

        // Setup WebSocket callbacks
        setupWebSocket();

        // Setup UI events
        setupUI();

        // Initialize click interaction handler
        await interaction.init();

        // Connect
        ws.connect();

        // Handle window resize
        window.addEventListener('resize', () => live2d.resize());
    }

    /**
     * Setup WebSocket event handlers.
     */
    function setupWebSocket() {
        ws.onConnected = () => {
            connectionDot.classList.add('connected');
            statusText.textContent = 'Connected';
            // Fetch cached interaction phrases from backend
            interaction.requestCachedPhrases();
        };

        ws.onDisconnected = () => {
            connectionDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        };

        ws.onMessage = (message) => {
            handleServerMessage(message);
        };

        ws.onError = () => {
            statusText.textContent = 'Connection error';
        };
    }

    /**
     * Handle messages from server.
     * @param {object} message - Parsed JSON message.
     */
    function handleServerMessage(message) {
        switch (message.type) {
            case 'connected':
                // Load Live2D model from server-provided path
                if (message.model_path) {
                    live2d.loadModel('/' + message.model_path);
                }
                // Store neutral expression name for auto-reset
                if (message.neutral_expression) {
                    live2d._neutralExpression = message.neutral_expression;
                }
                addSystemMessage(message.message || 'Connected!');
                break;

            case 'llm-response':
                addAssistantMessage(message.text);
                // Update emotion
                if (message.emotion) {
                    emotionBadge.textContent = message.emotion;
                    live2d.setEmotion(message.emotion, message.expression);
                }
                removeTypingIndicator();
                break;

            case 'tts-audio':
                if (message.audio) {
                    audio.queueAudio(message.audio);
                }
                break;

            case 'interaction-response':
                interaction.handleLlmResponse(message);
                break;

            case 'interaction-phrases':
                interaction.handlePhrasesResponse(message.phrases);
                break;

            case 'asr-result':
                if (message.text) {
                    addUserMessage(message.text);
                    showTypingIndicator();
                }
                break;

            case 'vad-status':
                if (message.status === 'speech_start') {
                    btnMic.classList.add('speaking');
                    console.log('[App] VAD: speech detected');
                } else if (message.status === 'speech_end') {
                    btnMic.classList.remove('speaking');
                    showTypingIndicator();
                    console.log('[App] VAD: speech ended, processing...');
                }
                break;

            case 'realtime-started':
                console.log('[App] Server confirmed real-time mode');
                break;

            case 'realtime-stopped':
                console.log('[App] Server confirmed real-time mode stopped');
                break;

            case 'pong':
                // Heartbeat response, ignore
                break;

            default:
                console.log('[App] Unknown message type:', message.type);
        }
    }

    /**
     * Setup UI event handlers.
     */
    function setupUI() {
        // Send text on button click
        btnSend.addEventListener('click', sendTextMessage);

        // Send text on Enter key
        textInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendTextMessage();
            }
        });

        // Microphone button - click to toggle recording
        btnMic.addEventListener('click', toggleRecording);
    }

    /**
     * Send text message from input field.
     */
    function sendTextMessage() {
        const text = textInput.value.trim();
        if (!text) return;

        addUserMessage(text);
        ws.sendText(text);
        textInput.value = '';

        showTypingIndicator();
    }

    // Real-time voice mode state
    let realtimeMode = false;
    let realtimeStream = null;
    let realtimeProcessor = null;

    /**
     * Toggle real-time voice mode.
     * Click once to start, click again to manually stop.
     * Server-side VAD will auto-detect speech endpoints.
     */
    async function toggleRecording() {
        if (realtimeMode) {
            // Stop real-time mode
            stopRealtimeMode();
        } else {
            // Start real-time mode
            await startRealtimeMode();
        }
    }

    // Keep references to prevent GC
    let realtimeSource = null;
    let rtContext = null;

    /**
     * Start real-time voice mode - continuous streaming with server-side VAD.
     * Uses AudioWorklet for reliable audio capture (ScriptProcessorNode produces
     * silence in modern Chromium with MediaStreamSource).
     */
    async function startRealtimeMode() {
        try {
            // Get microphone stream first (user gesture is active here)
            realtimeStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                },
            });

            // Diagnose the MediaStream track
            const track = realtimeStream.getAudioTracks()[0];
            console.log(`[Audio Diag] Track: label="${track.label}", enabled=${track.enabled}, muted=${track.muted}, readyState=${track.readyState}`);
            console.log(`[Audio Diag] Track settings:`, JSON.stringify(track.getSettings()));

            // Create a FRESH AudioContext on user click
            rtContext = new (window.AudioContext || window.webkitAudioContext)();
            await rtContext.resume();

            const nativeRate = rtContext.sampleRate;
            const targetRate = 16000;
            const ratio = nativeRate / targetRate;

            console.log(`[App] RT AudioContext: sampleRate=${nativeRate}, state=${rtContext.state}`);

            // Register AudioWorklet module
            await rtContext.audioWorklet.addModule('js/audio-processor.js');

            // Create source from the live microphone stream
            realtimeSource = rtContext.createMediaStreamSource(realtimeStream);

            // Diagnostic: use AnalyserNode to independently verify signal
            const analyser = rtContext.createAnalyser();
            analyser.fftSize = 2048;
            realtimeSource.connect(analyser);
            const diagBuffer = new Float32Array(analyser.fftSize);
            let diagCount = 0;
            const diagInterval = setInterval(() => {
                analyser.getFloatTimeDomainData(diagBuffer);
                let maxVal = 0;
                for (let i = 0; i < diagBuffer.length; i++) {
                    const abs = Math.abs(diagBuffer[i]);
                    if (abs > maxVal) maxVal = abs;
                }
                diagCount++;
                if (diagCount <= 5) {
                    console.log(`[Audio Diag] Analyser check #${diagCount}: max=${maxVal.toFixed(6)}`);
                }
                if (diagCount >= 5) clearInterval(diagInterval);
            }, 500);


            // Create AudioWorklet node
            const workletNode = new AudioWorkletNode(rtContext, 'audio-capture-processor');

            let _rtDebugCount = 0;
            workletNode.port.onmessage = (event) => {
                if (!realtimeMode) return;
                const inputData = event.data.buffer;

                // Debug: log raw audio stats for first 5 chunks
                _rtDebugCount++;
                if (_rtDebugCount <= 5) {
                    let maxVal = 0;
                    for (let i = 0; i < inputData.length; i++) {
                        const abs = Math.abs(inputData[i]);
                        if (abs > maxVal) maxVal = abs;
                    }
                    console.log(`[Audio Debug] Worklet chunk #${_rtDebugCount}: length=${inputData.length}, max=${maxVal.toFixed(6)}, state=${rtContext.state}`);
                }

                // Resample to 16kHz
                const newLength = Math.round(inputData.length / ratio);
                const resampled = new Float32Array(newLength);
                for (let i = 0; i < newLength; i++) {
                    const srcIdx = i * ratio;
                    const floor = Math.floor(srcIdx);
                    const ceil = Math.min(floor + 1, inputData.length - 1);
                    const frac = srcIdx - floor;
                    resampled[i] = inputData[floor] * (1 - frac) + inputData[ceil] * frac;
                }

                const chunk = Array.from(resampled);
                ws.sendRealtimeAudio(chunk);
            };

            // Connect: source -> worklet -> destination
            realtimeSource.connect(workletNode);
            workletNode.connect(rtContext.destination);

            // Store worklet node ref for cleanup
            realtimeProcessor = workletNode;

            // Signal server to start real-time mode
            ws.sendRealtimeStart();

            realtimeMode = true;
            btnMic.classList.add('recording');
            btnMic.title = 'Click to stop voice mode';
            console.log(`[App] Real-time voice mode started (AudioWorklet, native: ${nativeRate}Hz -> ${targetRate}Hz)`);

        } catch (e) {
            console.error('[App] Failed to start real-time mode:', e);
            addSystemMessage('Microphone access denied. Please allow microphone access.');
        }
    }

    /**
     * Stop real-time voice mode.
     */
    function stopRealtimeMode() {
        realtimeMode = false;

        // Clean up audio nodes
        if (realtimeProcessor) {
            realtimeProcessor.disconnect();
            realtimeProcessor = null;
        }

        if (realtimeSource) {
            realtimeSource.disconnect();
            realtimeSource = null;
        }

        // Close the dedicated RT AudioContext
        if (rtContext) {
            rtContext.close();
            rtContext = null;
        }

        if (realtimeStream) {
            realtimeStream.getTracks().forEach(track => track.stop());
            realtimeStream = null;
        }

        // Signal server to stop
        ws.sendRealtimeStop();

        btnMic.classList.remove('recording');
        btnMic.title = 'Click to start voice mode';
        console.log('[App] Real-time voice mode stopped');
    }

    /**
     * Add a user message bubble to chat.
     * @param {string} text - Message text.
     */
    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message message-user';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    /**
     * Add an assistant message bubble to chat.
     * @param {string} text - Message text.
     */
    function addAssistantMessage(text) {
        const div = document.createElement('div');
        div.className = 'message message-assistant';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    /**
     * Add a system message to chat.
     * @param {string} text - System message text.
     */
    function addSystemMessage(text) {
        const div = document.createElement('div');
        div.className = 'message message-system';
        div.textContent = text;
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    /**
     * Show typing indicator.
     */
    function showTypingIndicator() {
        removeTypingIndicator();
        const div = document.createElement('div');
        div.className = 'typing-indicator';
        div.id = 'typingIndicator';
        div.innerHTML = '<span></span><span></span><span></span>';
        chatMessages.appendChild(div);
        scrollToBottom();
    }

    /**
     * Remove typing indicator.
     */
    function removeTypingIndicator() {
        const existing = document.getElementById('typingIndicator');
        if (existing) existing.remove();
    }

    /**
     * Scroll chat to bottom.
     */
    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Start the app
    document.addEventListener('DOMContentLoaded', init);
})();
