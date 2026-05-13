/**
 * Live2D model rendering and control for Mimosa.
 * Uses pixi-live2d-display for model loading and expression/motion control.
 */
class Live2DManager {
    constructor(canvasId) {
        this.canvasId = canvasId;
        this.app = null;
        this.model = null;
        this.isLoaded = false;

        // Emotion to icon mapping
        this.emotionIcons = {
            neutral: '😊',
            joy: '😄',
            surprise: '😲',
            sadness: '😢',
            anger: '😠',
            fear: '😨',
            disgust: '😖',
            love: '🥰',
        };
    }

    /**
     * Initialize the PIXI application and load model.
     * @param {string} modelPath - Relative path to model3.json file.
     */
    async init(modelPath) {
        const canvas = document.getElementById(this.canvasId);
        if (!canvas) {
            console.error('[Live2D] Canvas element not found:', this.canvasId);
            return false;
        }

        try {
            // Initialize PIXI Application
            this.app = new PIXI.Application({
                view: canvas,
                autoStart: true,
                resizeTo: canvas.parentElement,
                backgroundAlpha: 0,
            });

            if (modelPath) {
                await this.loadModel(modelPath);
            }

            return true;
        } catch (e) {
            console.error('[Live2D] Initialization failed:', e);
            return false;
        }
    }

    /**
     * Load a Live2D model.
     * @param {string} modelPath - URL path to model3.json.
     */
    async loadModel(modelPath) {
        try {
            // Remove existing model
            if (this.model) {
                this.app.stage.removeChild(this.model);
                this.model.destroy();
            }

            console.log('[Live2D] Loading model:', modelPath);

            // Load model using pixi-live2d-display
            this.model = await PIXI.live2d.Live2DModel.from(modelPath);

            // Scale and position
            const scale = Math.min(
                this.app.screen.width / this.model.width,
                this.app.screen.height / this.model.height
            ) * 0.7;

            this.model.scale.set(scale);
            this.model.x = this.app.screen.width / 2;
            this.model.y = this.app.screen.height / 2;
            this.model.anchor.set(0.5, 0.5);

            // Enable mouse tracking (look at cursor)
            this.model.interactive = true;
            this.model.trackedPointers = {};

            // Add to stage
            this.app.stage.addChild(this.model);

            this.isLoaded = true;
            console.log('[Live2D] Model loaded successfully');

            // Start idle motion
            this._playIdleMotion();

            return true;
        } catch (e) {
            console.error('[Live2D] Failed to load model:', e);
            this.isLoaded = false;
            return false;
        }
    }

    /**
     * Set expression by name.
     * @param {string} expressionName - Expression name (e.g., 'exp_01').
     */
    setExpression(expressionName) {
        if (!this.model || !this.isLoaded) return;

        try {
            this.model.expression(expressionName);
            console.log('[Live2D] Expression set:', expressionName);
        } catch (e) {
            console.warn('[Live2D] Failed to set expression:', expressionName, e);
        }
    }

    /**
     * Play a motion by group and index.
     * @param {string} group - Motion group name.
     * @param {number} index - Motion index in group.
     * @param {number} [priority=3] - Motion priority (3=FORCE, interrupts current motion).
     */
    playMotion(group = '', index = 0, priority = 3) {
        if (!this.model || !this.isLoaded) return;

        try {
            this.model.motion(group, index, priority);
            console.log('[Live2D] Motion played:', group, index, 'priority:', priority);
        } catch (e) {
            console.warn('[Live2D] Failed to play motion:', e);
        }
    }

    /**
     * Set expression based on emotion tag.
     * @param {string} emotion - Emotion name (e.g., 'joy', 'sadness').
     * @param {string} expressionName - Expression file name from model.
     */
    setEmotion(emotion, expressionName) {
        if (expressionName) {
            this.setExpression(expressionName);
        }

        // Play a random motion for liveliness
        if (emotion !== 'neutral') {
            this.playMotion('', Math.floor(Math.random() * 3));
        }

        // Update emotion indicator
        const indicator = document.getElementById('emotionIndicator');
        if (indicator) {
            const icon = this.emotionIcons[emotion] || '😊';
            indicator.querySelector('.emotion-icon').textContent = icon;
        }
    }

    /**
     * Play idle motion loop.
     */
    _playIdleMotion() {
        if (!this.model || !this.isLoaded) return;

        try {
            this.model.motion('Idle', 0);
        } catch (e) {
            // Idle motion may not exist, ignore
        }
    }

    /**
     * Handle window resize.
     */
    resize() {
        if (!this.app || !this.model) return;

        const scale = Math.min(
            this.app.screen.width / this.model.width,
            this.app.screen.height / this.model.height
        ) * 0.7;

        this.model.scale.set(scale);
        this.model.x = this.app.screen.width / 2;
        this.model.y = this.app.screen.height / 2;
    }
}
