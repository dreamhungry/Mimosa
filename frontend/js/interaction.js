/**
 * Click interaction handler for Mimosa Live2D character.
 * Provides instant random reactions (emoji, motion, speech bubble)
 * with configurable probability to trigger LLM-generated responses.
 *
 * Configuration is loaded from js/interaction-config.json at init time.
 */
class InteractionManager {
    /**
     * @param {Live2DManager} live2d - Live2D manager instance.
     * @param {MimosaWebSocket} ws - WebSocket client instance.
     */
    constructor(live2d, ws) {
        this.live2d = live2d;
        this.ws = ws;

        // Internal state
        this._clickCount = 0;
        this._lastClickTime = 0;
        this._bubbleTimer = null;
        this._bubbleEl = null;
        this._waitingLlm = false;

        // Defaults (overridden by config file)
        this.cooldownMs = 2000;
        this.llmThreshold = 3;
        this.llmProbability = 0.5;
        this.bubbleDurationMs = 3000;
        this._phrases = [];
        this._emojis = [];
        this._motionIndices = [0, 1, 2, 3, 4, 5];

        // LLM-generated phrase cache (synced from backend)
        this._cachedPhrases = [];
    }

    /**
     * Load config from JSON file and initialize click handler.
     * We bind to `.live2d-area` instead of the canvas because
     * PIXI's interaction system captures pointer events on the canvas.
     */
    async init() {
        // Load external config
        await this._loadConfig();

        const area = document.querySelector('.live2d-area');
        if (!area) {
            console.warn('[Interaction] .live2d-area not found');
            return;
        }

        this._createBubbleElement();

        area.addEventListener('click', (e) => this._handleClick(e));
        console.log('[Interaction] Click handler initialized on .live2d-area');
    }

    /**
     * Fetch interaction config from external JSON file.
     */
    async _loadConfig() {
        try {
            const resp = await fetch('js/interaction-config.json');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const cfg = await resp.json();

            if (cfg.cooldownMs) this.cooldownMs = cfg.cooldownMs;
            if (cfg.llmThreshold) this.llmThreshold = cfg.llmThreshold;
            if (cfg.llmProbability) this.llmProbability = cfg.llmProbability;
            if (cfg.bubbleDurationMs) this.bubbleDurationMs = cfg.bubbleDurationMs;
            if (cfg.phrases) this._phrases = cfg.phrases;
            if (cfg.emojis) this._emojis = cfg.emojis;
            if (cfg.motionIndices) this._motionIndices = cfg.motionIndices;

            console.log('[Interaction] Config loaded:', this._phrases.length, 'phrases,', this._emojis.length, 'emojis');
        } catch (e) {
            console.warn('[Interaction] Failed to load config, using defaults:', e.message);
            // Fallback defaults
            this._phrases = ['Hi~', 'Hey!', 'Nyaa~'];
            this._emojis = ['\u2728', '\ud83d\ude0a', '\ud83d\udc96'];
        }
    }

    /**
     * Request cached LLM phrases from backend.
     * Called after WebSocket connection is established.
     */
    requestCachedPhrases() {
        if (!this.ws || !this.ws.isConnected) return;
        this.ws.send({ type: 'get-interaction-phrases' });
        console.log('[Interaction] Requesting cached phrases from backend');
    }

    /**
     * Handle the phrase pool response from backend.
     * @param {string[]} phrases - Array of cached phrases.
     */
    handlePhrasesResponse(phrases) {
        if (Array.isArray(phrases)) {
            this._cachedPhrases = phrases;
            console.log('[Interaction] Loaded', this._cachedPhrases.length, 'cached phrases from backend');
        }
    }

    /**
     * Handle LLM interaction response from server.
     * The backend persists the phrase; we just update our in-memory pool.
     * @param {object} message - Server message with text, emotion, expression.
     */
    handleLlmResponse(message) {
        this._waitingLlm = false;
        if (message.text) {
            this._showBubble(message.text);
            // Add to in-memory pool (backend already persisted it)
            if (!this._cachedPhrases.includes(message.text)) {
                this._cachedPhrases.push(message.text);
            }
        }
        if (message.emotion) {
            this.live2d.setEmotion(message.emotion, message.expression);
            const badge = document.getElementById('emotionBadge');
            if (badge) badge.textContent = message.emotion;
        }
    }

    /**
     * Create the speech bubble DOM element.
     */
    _createBubbleElement() {
        if (this._bubbleEl) return;

        const bubble = document.createElement('div');
        bubble.className = 'interaction-bubble';
        bubble.style.display = 'none';

        const area = document.querySelector('.live2d-area');
        if (area) {
            area.appendChild(bubble);
        }
        this._bubbleEl = bubble;
    }

    /**
     * Handle a click on the Live2D area.
     * @param {MouseEvent} e - Click event.
     */
    _handleClick(e) {
        const now = Date.now();

        // Cooldown check
        if (now - this._lastClickTime < this.cooldownMs) {
            return;
        }
        this._lastClickTime = now;
        this._clickCount++;

        console.log(`[Interaction] Click #${this._clickCount}`);

        // Show click ripple effect
        this._showRipple(e);

        // Check if we should trigger LLM
        let llmTriggered = false;
        if (this._clickCount >= this.llmThreshold && !this._waitingLlm) {
            if (Math.random() < this.llmProbability) {
                this._triggerLlmInteraction();
                this._clickCount = 0; // Reset counter after LLM trigger
                llmTriggered = true;
            }
        }

        if (llmTriggered) {
            // LLM triggered: only play motion (no bubble text to avoid double display)
            this._playMotionOnly();
        } else {
            // Normal click: full instant local reaction with bubble
            this._playInstantReaction();
        }
    }

    /**
     * Play only a random motion without showing a speech bubble.
     * Used when LLM interaction is triggered to avoid double-display.
     * Priority=2 (NORMAL) can interrupt idle motions while preserving expression
     * (pixi-live2d-display defaults preserveExpressionOnMotion=true).
     */
    _playMotionOnly() {
        const motionIdx = this._motionIndices[
            Math.floor(Math.random() * this._motionIndices.length)
        ];
        this.live2d.playMotion('', motionIdx, 2);
    }

    /**
     * Play an instant local reaction (no server round-trip).
     * Phrases are drawn from the combined pool (static config + LLM cache).
     */
    _playInstantReaction() {
        // 1. Random motion (priority=2/NORMAL so it can interrupt idle)
        const motionIdx = this._motionIndices[
            Math.floor(Math.random() * this._motionIndices.length)
        ];
        this.live2d.playMotion('', motionIdx, 2);

        // 2. Random emoji on the emotion indicator
        const emoji = this._emojis[Math.floor(Math.random() * this._emojis.length)];
        const indicator = document.getElementById('emotionIndicator');
        if (indicator) {
            const icon = indicator.querySelector('.emotion-icon');
            if (icon) {
                icon.textContent = emoji;
                // Brief scale animation
                indicator.style.transform = 'scale(1.3)';
                setTimeout(() => {
                    indicator.style.transform = 'scale(1)';
                }, 200);
            }
        }

        // 3. Show random speech bubble from combined pool
        const allPhrases = [...this._phrases, ...this._cachedPhrases];
        const phrase = allPhrases[Math.floor(Math.random() * allPhrases.length)];
        this._showBubble(phrase);
    }

    /**
     * Show a speech bubble above the model.
     * @param {string} text - Text to display.
     */
    _showBubble(text) {
        if (!this._bubbleEl) return;

        // Clear existing timer
        if (this._bubbleTimer) {
            clearTimeout(this._bubbleTimer);
        }

        this._bubbleEl.textContent = text;
        this._bubbleEl.style.display = 'block';
        this._bubbleEl.classList.remove('fade-out');
        this._bubbleEl.classList.add('fade-in');

        this._bubbleTimer = setTimeout(() => {
            this._bubbleEl.classList.remove('fade-in');
            this._bubbleEl.classList.add('fade-out');
            setTimeout(() => {
                this._bubbleEl.style.display = 'none';
                this._bubbleEl.classList.remove('fade-out');
            }, 300);
        }, this.bubbleDurationMs);
    }

    /**
     * Show a ripple effect at the click position.
     * @param {MouseEvent} e - Click event.
     */
    _showRipple(e) {
        const area = document.querySelector('.live2d-area');
        if (!area) return;

        const rect = area.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        const ripple = document.createElement('div');
        ripple.className = 'click-ripple';
        ripple.style.left = x + 'px';
        ripple.style.top = y + 'px';
        area.appendChild(ripple);

        // Remove after animation
        ripple.addEventListener('animationend', () => ripple.remove());
    }

    /**
     * Send an interaction request to the server for LLM-generated response.
     */
    _triggerLlmInteraction() {
        if (!this.ws || !this.ws.isConnected) return;

        this._waitingLlm = true;
        console.log('[Interaction] Triggering LLM interaction');

        this.ws.send({
            type: 'interaction',
            trigger: 'click',
            click_count: this._clickCount,
        });
    }
}
