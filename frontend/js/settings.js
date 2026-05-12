/**
 * Settings panel for Mimosa.
 * Provides a modal with tabs: Character (personality radar + sliders)
 * and LLM configuration.
 */
console.log('[Settings] settings.js loaded');

// eslint-disable-next-line no-unused-vars
const SettingsPanel = (function () {
    'use strict';

    // ---------------------------------------------------------------
    // Constants
    // ---------------------------------------------------------------
    const BIG_FIVE_TRAITS = [
        { key: 'openness', label: 'Openness' },
        { key: 'conscientiousness', label: 'Conscientiousness' },
        { key: 'extraversion', label: 'Extraversion' },
        { key: 'agreeableness', label: 'Agreeableness' },
        { key: 'neuroticism', label: 'Neuroticism' },
    ];

    const HUMOR_STYLES = ['gentle', 'witty', 'sarcastic', 'dry'];

    // ---------------------------------------------------------------
    // State
    // ---------------------------------------------------------------
    let currentTab = 'character';
    let personalityData = null;
    let llmData = null;
    let modalEl = null;

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    /**
     * Initialize settings panel - inject button and modal into DOM.
     */
    function init() {
        injectSettingsButton();
        injectModal();
        console.log('[Settings] initialized');
    }

    // ---------------------------------------------------------------
    // DOM Injection
    // ---------------------------------------------------------------

    /** Add gear button to status bar. */
    function injectSettingsButton() {
        const statusLeft = document.querySelector('.status-left');
        if (!statusLeft) return;

        const btn = document.createElement('button');
        btn.className = 'btn-settings';
        btn.id = 'btnSettings';
        btn.title = 'Settings';
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none"
                 stroke="currentColor" stroke-width="2" stroke-linecap="round"
                 stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1
                    -2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65
                    0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9
                    19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83
                    -2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0
                    0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6
                    9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83
                    -2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0
                    0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1
                    1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1
                    2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65
                    1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65
                    1.65 0 0 0-1.51 1z"/>
            </svg>`;
        btn.addEventListener('click', openModal);
        statusLeft.appendChild(btn);
    }

    /** Create the modal skeleton (hidden by default). */
    function injectModal() {
        modalEl = document.createElement('div');
        modalEl.className = 'settings-overlay';
        modalEl.id = 'settingsOverlay';
        modalEl.innerHTML = `
            <div class="settings-modal">
                <div class="settings-header">
                    <h2 class="settings-title">Settings</h2>
                    <button class="settings-close" id="settingsClose">&times;</button>
                </div>
                <div class="settings-body">
                    <nav class="settings-nav">
                        <button class="settings-tab active" data-tab="character">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                                 stroke="currentColor" stroke-width="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
                                <circle cx="12" cy="7" r="4"/>
                            </svg>
                            Character
                        </button>
                        <button class="settings-tab" data-tab="llm">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                                 stroke="currentColor" stroke-width="2">
                                <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/>
                                <line x1="8" y1="21" x2="16" y2="21"/>
                                <line x1="12" y1="17" x2="12" y2="21"/>
                            </svg>
                            LLM
                        </button>
                    </nav>
                    <div class="settings-content" id="settingsContent"></div>
                </div>
            </div>`;
        document.body.appendChild(modalEl);

        // Event: close button
        document.getElementById('settingsClose').addEventListener('click', closeModal);

        // Event: click overlay backdrop
        modalEl.addEventListener('click', (e) => {
            if (e.target === modalEl) closeModal();
        });

        // Event: tab switching
        modalEl.querySelectorAll('.settings-tab').forEach((tab) => {
            tab.addEventListener('click', () => switchTab(tab.dataset.tab));
        });

        // Event: Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modalEl.classList.contains('open')) {
                closeModal();
            }
        });
    }

    // ---------------------------------------------------------------
    // Modal open / close
    // ---------------------------------------------------------------

    async function openModal() {
        modalEl.classList.add('open');
        currentTab = 'character';
        highlightTab('character');
        await loadCharacterTab();
    }

    function closeModal() {
        modalEl.classList.remove('open');
    }

    // ---------------------------------------------------------------
    // Tab switching
    // ---------------------------------------------------------------

    function highlightTab(tabId) {
        modalEl.querySelectorAll('.settings-tab').forEach((t) => {
            t.classList.toggle('active', t.dataset.tab === tabId);
        });
    }

    async function switchTab(tabId) {
        if (tabId === currentTab) return;
        currentTab = tabId;
        highlightTab(tabId);

        // Re-trigger fade-in animation on content area
        const content = document.getElementById('settingsContent');
        content.style.animation = 'none';
        // Force reflow to restart animation
        void content.offsetHeight;
        content.style.animation = '';

        if (tabId === 'character') {
            await loadCharacterTab();
        } else if (tabId === 'llm') {
            await loadLLMTab();
        }
    }

    // ---------------------------------------------------------------
    // Character Tab
    // ---------------------------------------------------------------

    async function loadCharacterTab() {
        const content = document.getElementById('settingsContent');
        content.innerHTML = '<div class="settings-loading">Loading...</div>';

        try {
            const res = await fetch('/api/personality');
            const json = await res.json();
            personalityData = json.data;
        } catch (e) {
            content.innerHTML = '<div class="settings-error">Failed to load personality data.</div>';
            return;
        }

        renderCharacterTab();
    }

    function renderCharacterTab() {
        const content = document.getElementById('settingsContent');
        const bf = personalityData.big_five;
        const style = personalityData.style;

        content.innerHTML = `
            <div class="character-tab">
                <div class="radar-section">
                    <canvas id="radarCanvas" width="240" height="240"></canvas>
                </div>
                <div class="sliders-section">
                    <h3 class="section-label">Big Five Traits</h3>
                    ${BIG_FIVE_TRAITS.map((t) => `
                        <div class="slider-row">
                            <label class="slider-label">${t.label}</label>
                            <input type="range" class="slider-input" id="slider-${t.key}"
                                   min="0" max="100" value="${bf[t.key]}"
                                   data-trait="${t.key}">
                            <span class="slider-value" id="val-${t.key}">${bf[t.key]}</span>
                        </div>
                    `).join('')}

                    <h3 class="section-label" style="margin-top:16px">Style</h3>
                    <div class="slider-row">
                        <label class="slider-label">Humor</label>
                        <select class="select-input" id="select-humor">
                            ${HUMOR_STYLES.map((s) =>
                                `<option value="${s}" ${s === style.humor_style ? 'selected' : ''}>${s}</option>`
                            ).join('')}
                        </select>
                    </div>
                    <div class="slider-row">
                        <label class="slider-label">Formality</label>
                        <input type="range" class="slider-input" id="slider-formality"
                               min="0" max="100" value="${style.speech_formality}">
                        <span class="slider-value" id="val-formality">${style.speech_formality}</span>
                    </div>

                    <div class="btn-group">
                        <button class="btn-action btn-save" id="btnSavePersonality">Save</button>
                        <button class="btn-action btn-reset" id="btnResetPersonality">Reset</button>
                    </div>
                </div>
            </div>`;

        // Draw initial radar
        drawRadar();

        // Bind slider events
        BIG_FIVE_TRAITS.forEach((t) => {
            const slider = document.getElementById(`slider-${t.key}`);
            slider.addEventListener('input', () => {
                document.getElementById(`val-${t.key}`).textContent = slider.value;
                drawRadar();
            });
        });

        const formalitySlider = document.getElementById('slider-formality');
        formalitySlider.addEventListener('input', () => {
            document.getElementById('val-formality').textContent = formalitySlider.value;
        });

        // Save button
        document.getElementById('btnSavePersonality').addEventListener('click', savePersonality);

        // Reset button
        document.getElementById('btnResetPersonality').addEventListener('click', resetPersonality);
    }

    async function savePersonality() {
        const bigFive = {};
        BIG_FIVE_TRAITS.forEach((t) => {
            bigFive[t.key] = parseInt(document.getElementById(`slider-${t.key}`).value, 10);
        });

        const style = {
            humor_style: document.getElementById('select-humor').value,
            speech_formality: parseInt(document.getElementById('slider-formality').value, 10),
        };

        const btn = document.getElementById('btnSavePersonality');
        btn.disabled = true;
        btn.textContent = 'Saving...';

        try {
            const res = await fetch('/api/personality', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ big_five: bigFive, style }),
            });
            const json = await res.json();
            if (json.status === 'ok') {
                personalityData = json.data;
                btn.textContent = 'Saved!';
                setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 1200);
            } else {
                throw new Error(json.detail || 'Save failed');
            }
        } catch (e) {
            btn.textContent = 'Error';
            setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 1500);
            console.error('[Settings] Save personality error:', e);
        }
    }

    async function resetPersonality() {
        const btn = document.getElementById('btnResetPersonality');
        btn.disabled = true;
        btn.textContent = 'Resetting...';

        try {
            const res = await fetch('/api/personality/reset', { method: 'POST' });
            const json = await res.json();
            if (json.status === 'ok') {
                personalityData = json.data;
                renderCharacterTab();
            }
        } catch (e) {
            btn.textContent = 'Error';
            setTimeout(() => { btn.textContent = 'Reset'; btn.disabled = false; }, 1500);
            console.error('[Settings] Reset personality error:', e);
        }
    }

    // ---------------------------------------------------------------
    // Radar Chart (Canvas)
    // ---------------------------------------------------------------

    function drawRadar() {
        const canvas = document.getElementById('radarCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        const w = 240;
        const h = 240;

        canvas.width = w * dpr;
        canvas.height = h * dpr;
        canvas.style.width = w + 'px';
        canvas.style.height = h + 'px';
        ctx.scale(dpr, dpr);

        const cx = w / 2;
        const cy = h / 2;
        const maxR = 90;
        const n = BIG_FIVE_TRAITS.length;
        const angleStep = (2 * Math.PI) / n;
        const startAngle = -Math.PI / 2;

        ctx.clearRect(0, 0, w, h);

        // Read current slider values
        const values = BIG_FIVE_TRAITS.map((t) => {
            const el = document.getElementById(`slider-${t.key}`);
            return el ? parseInt(el.value, 10) : 50;
        });

        // Draw grid rings
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.15)';
        ctx.lineWidth = 1;
        for (let ring = 1; ring <= 4; ring++) {
            const r = (maxR / 4) * ring;
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const angle = startAngle + angleStep * (i % n);
                const x = cx + r * Math.cos(angle);
                const y = cy + r * Math.sin(angle);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.closePath();
            ctx.stroke();
        }

        // Draw axis lines
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.12)';
        for (let i = 0; i < n; i++) {
            const angle = startAngle + angleStep * i;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(cx + maxR * Math.cos(angle), cy + maxR * Math.sin(angle));
            ctx.stroke();
        }

        // Draw data polygon (filled)
        ctx.beginPath();
        values.forEach((v, i) => {
            const r = (v / 100) * maxR;
            const angle = startAngle + angleStep * i;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(139, 92, 246, 0.25)';
        ctx.fill();
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.8)';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw data points
        values.forEach((v, i) => {
            const r = (v / 100) * maxR;
            const angle = startAngle + angleStep * i;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#8B5CF6';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1.5;
            ctx.stroke();
        });

        // Draw labels
        ctx.fillStyle = '#94A3B8';
        ctx.font = '11px Poppins, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        BIG_FIVE_TRAITS.forEach((t, i) => {
            const angle = startAngle + angleStep * i;
            const labelR = maxR + 18;
            const x = cx + labelR * Math.cos(angle);
            const y = cy + labelR * Math.sin(angle);
            // Abbreviate labels for compact display
            const abbr = t.label.substring(0, 1);
            ctx.fillText(abbr, x, y);
        });
    }

    // ---------------------------------------------------------------
    // LLM Tab
    // ---------------------------------------------------------------

    async function loadLLMTab() {
        const content = document.getElementById('settingsContent');
        content.innerHTML = '<div class="settings-loading">Loading...</div>';

        try {
            const res = await fetch('/api/config/llm');
            const json = await res.json();
            llmData = json.data;
        } catch (e) {
            content.innerHTML = '<div class="settings-error">Failed to load LLM config.</div>';
            return;
        }

        renderLLMTab();
    }

    function renderLLMTab() {
        const content = document.getElementById('settingsContent');
        const d = llmData;

        content.innerHTML = `
            <div class="llm-tab">
                <h3 class="section-label">LLM Configuration</h3>

                <div class="form-row">
                    <label class="form-label">Provider</label>
                    <input type="text" class="form-input" id="llm-provider"
                           value="${d.provider}" disabled>
                </div>

                <div class="form-row">
                    <label class="form-label">Model</label>
                    <input type="text" class="form-input" id="llm-model"
                           value="${d.model}" placeholder="e.g. gpt-4o-mini">
                </div>

                <div class="form-row">
                    <label class="form-label">Base URL</label>
                    <input type="text" class="form-input" id="llm-base-url"
                           value="${d.base_url}" disabled>
                </div>

                <div class="form-row">
                    <label class="form-label">API Key</label>
                    <input type="password" class="form-input" id="llm-api-key"
                           value="" placeholder="${d.api_key}">
                </div>

                <div class="form-row">
                    <label class="form-label">Temperature</label>
                    <input type="range" class="slider-input" id="llm-temperature"
                           min="0" max="200" value="${Math.round(d.temperature * 100)}">
                    <span class="slider-value" id="val-temperature">${d.temperature}</span>
                </div>

                <div class="form-row">
                    <label class="form-label">Max Tokens</label>
                    <input type="number" class="form-input form-input-sm" id="llm-max-tokens"
                           value="${d.max_tokens}" min="1" max="32768">
                </div>

                <div class="btn-group">
                    <button class="btn-action btn-save" id="btnSaveLLM">Save</button>
                </div>
            </div>`;

        // Temperature slider live update
        const tempSlider = document.getElementById('llm-temperature');
        tempSlider.addEventListener('input', () => {
            document.getElementById('val-temperature').textContent =
                (parseInt(tempSlider.value, 10) / 100).toFixed(2);
        });

        // Save button
        document.getElementById('btnSaveLLM').addEventListener('click', saveLLMConfig);
    }

    async function saveLLMConfig() {
        const body = {};

        const model = document.getElementById('llm-model').value.trim();
        if (model && model !== llmData.model) body.model = model;

        const tempVal = parseInt(document.getElementById('llm-temperature').value, 10) / 100;
        if (tempVal !== llmData.temperature) body.temperature = tempVal;

        const maxTokens = parseInt(document.getElementById('llm-max-tokens').value, 10);
        if (maxTokens && maxTokens !== llmData.max_tokens) body.max_tokens = maxTokens;

        const apiKey = document.getElementById('llm-api-key').value.trim();
        if (apiKey) body.api_key = apiKey;

        if (Object.keys(body).length === 0) return; // Nothing changed

        const btn = document.getElementById('btnSaveLLM');
        btn.disabled = true;
        btn.textContent = 'Saving...';

        try {
            const res = await fetch('/api/config/llm', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            const json = await res.json();
            if (json.status === 'ok') {
                llmData = json.data;
                btn.textContent = 'Saved!';
                setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 1200);
            } else {
                throw new Error(json.detail || 'Save failed');
            }
        } catch (e) {
            btn.textContent = 'Error';
            setTimeout(() => { btn.textContent = 'Save'; btn.disabled = false; }, 1500);
            console.error('[Settings] Save LLM config error:', e);
        }
    }

    // ---------------------------------------------------------------
    // Bootstrap
    // ---------------------------------------------------------------
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // DOM already parsed, run immediately
        init();
    }

    return { open: openModal, close: closeModal };
})();
