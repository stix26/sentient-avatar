// Sentient Avatar SillyTavern Extension
// This extension adds real-time avatar support to SillyTavern

class SentientAvatarExtension {
    constructor() {
        this.avatarContainer = null;
        this.audioContext = null;
        this.mediaRecorder = null;
        this.websocket = null;
        this.isRecording = false;
        this.isPlaying = false;
        this.currentAudio = null;
        this.currentVideo = null;
        this.voice = null;
        this.model = "bark";
        this.temperature = 0.7;
        this.speed = 1.0;
        this.animationStyle = "natural";
        this.motionScale = 1.0;
        this.stillness = false;
        this.referenceImage = null;
    }

    // Initialize extension
    async init() {
        // Create avatar container
        this.avatarContainer = document.createElement('div');
        this.avatarContainer.id = 'sentient-avatar-container';
        this.avatarContainer.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 320px;
            height: 240px;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            z-index: 9999;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;

        // Create video element
        this.videoElement = document.createElement('video');
        this.videoElement.style.cssText = `
            width: 100%;
            height: 100%;
            object-fit: cover;
        `;
        this.avatarContainer.appendChild(this.videoElement);

        // Create controls
        const controls = document.createElement('div');
        controls.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 10px;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            justify-content: space-between;
            align-items: center;
        `;

        // Add record button
        this.recordButton = document.createElement('button');
        this.recordButton.textContent = '🎤';
        this.recordButton.style.cssText = `
            background: none;
            border: none;
            color: white;
            font-size: 24px;
            cursor: pointer;
            padding: 5px;
        `;
        this.recordButton.onclick = () => this.toggleRecording();
        controls.appendChild(this.recordButton);

        // Add settings button
        const settingsButton = document.createElement('button');
        settingsButton.textContent = '⚙️';
        settingsButton.style.cssText = this.recordButton.style.cssText;
        settingsButton.onclick = () => this.showSettings();
        controls.appendChild(settingsButton);

        this.avatarContainer.appendChild(controls);
        document.body.appendChild(this.avatarContainer);

        // Initialize audio context
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();

        // Connect to WebSocket
        this.connectWebSocket();

        // Add extension to SillyTavern
        this.addToSillyTavern();
    }

    // Connect to WebSocket
    connectWebSocket() {
        const clientId = Math.random().toString(36).substring(7);
        this.websocket = new WebSocket(`ws://localhost:8000/ws/${clientId}`);

        this.websocket.onopen = () => {
            console.log('Connected to Sentient Avatar WebSocket');
        };

        this.websocket.onmessage = async (event) => {
            const data = JSON.parse(event.data);
            await this.handleWebSocketMessage(data);
        };

        this.websocket.onclose = () => {
            console.log('Disconnected from Sentient Avatar WebSocket');
            // Try to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }

    // Handle WebSocket messages
    async handleWebSocketMessage(data) {
        switch (data.type) {
            case 'response':
                await this.playResponse(data);
                break;
            case 'image_analysis':
                this.handleImageAnalysis(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }

    // Play response
    async playResponse(data) {
        try {
            // Stop current playback
            if (this.isPlaying) {
                await this.stopPlayback();
            }

            this.isPlaying = true;

            // Play audio
            const audioData = base64ToArrayBuffer(data.audio);
            const audioBuffer = await this.audioContext.decodeAudioData(audioData);
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            source.start();
            this.currentAudio = source;

            // Play video
            const videoData = base64ToArrayBuffer(data.video);
            const videoBlob = new Blob([videoData], { type: 'video/mp4' });
            const videoUrl = URL.createObjectURL(videoBlob);
            this.videoElement.src = videoUrl;
            this.videoElement.play();
            this.currentVideo = this.videoElement;

            // Wait for playback to finish
            source.onended = () => {
                this.isPlaying = false;
                this.currentAudio = null;
                this.currentVideo = null;
            };

        } catch (error) {
            console.error('Error playing response:', error);
            this.isPlaying = false;
        }
    }

    // Stop playback
    async stopPlayback() {
        if (this.currentAudio) {
            this.currentAudio.stop();
            this.currentAudio = null;
        }
        if (this.currentVideo) {
            this.currentVideo.pause();
            this.currentVideo.src = '';
            this.currentVideo = null;
        }
        this.isPlaying = false;
    }

    // Toggle recording
    async toggleRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    // Start recording
    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.isRecording = true;
            this.recordButton.textContent = '⏹️';

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.sendAudioChunk(event.data);
                }
            };

            this.mediaRecorder.start(100); // Collect data every 100ms

        } catch (error) {
            console.error('Error starting recording:', error);
        }
    }

    // Stop recording
    async stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            this.recordButton.textContent = '🎤';

            // Stop all tracks
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    // Send audio chunk
    async sendAudioChunk(chunk) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const reader = new FileReader();
            reader.onload = () => {
                const base64data = reader.result.split(',')[1];
                this.websocket.send(JSON.stringify({
                    type: 'audio',
                    data: base64data,
                    is_final: !this.isRecording
                }));
            };
            reader.readAsDataURL(chunk);
        }
    }

    // Show settings
    showSettings() {
        // Create settings modal
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 10000;
        `;

        // Add settings form
        const form = document.createElement('form');
        form.innerHTML = `
            <h3>Avatar Settings</h3>
            <div>
                <label>Voice:</label>
                <select id="avatar-voice"></select>
            </div>
            <div>
                <label>Model:</label>
                <select id="avatar-model">
                    <option value="bark">Bark</option>
                    <option value="xtts">XTTS</option>
                </select>
            </div>
            <div>
                <label>Temperature:</label>
                <input type="range" id="avatar-temperature" min="0" max="1" step="0.1" value="${this.temperature}">
            </div>
            <div>
                <label>Speed:</label>
                <input type="range" id="avatar-speed" min="0.5" max="2" step="0.1" value="${this.speed}">
            </div>
            <div>
                <label>Animation Style:</label>
                <select id="avatar-style"></select>
            </div>
            <div>
                <label>Motion Scale:</label>
                <input type="range" id="avatar-motion" min="0" max="2" step="0.1" value="${this.motionScale}">
            </div>
            <div>
                <label>Stillness:</label>
                <input type="checkbox" id="avatar-stillness" ${this.stillness ? 'checked' : ''}>
            </div>
            <div>
                <label>Reference Image:</label>
                <input type="file" id="avatar-reference" accept="image/*">
            </div>
            <button type="submit">Save</button>
            <button type="button" onclick="this.closest('.modal').remove()">Cancel</button>
        `;

        // Load voices
        this.loadVoices().then(voices => {
            const voiceSelect = form.querySelector('#avatar-voice');
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.id;
                option.textContent = voice.name;
                voiceSelect.appendChild(option);
            });
            if (this.voice) {
                voiceSelect.value = this.voice;
            }
        });

        // Load styles
        this.loadStyles().then(styles => {
            const styleSelect = form.querySelector('#avatar-style');
            styles.forEach(style => {
                const option = document.createElement('option');
                option.value = style.id;
                option.textContent = style.name;
                styleSelect.appendChild(option);
            });
            if (this.animationStyle) {
                styleSelect.value = this.animationStyle;
            }
        });

        // Handle form submission
        form.onsubmit = (event) => {
            event.preventDefault();
            this.voice = form.querySelector('#avatar-voice').value;
            this.model = form.querySelector('#avatar-model').value;
            this.temperature = parseFloat(form.querySelector('#avatar-temperature').value);
            this.speed = parseFloat(form.querySelector('#avatar-speed').value);
            this.animationStyle = form.querySelector('#avatar-style').value;
            this.motionScale = parseFloat(form.querySelector('#avatar-motion').value);
            this.stillness = form.querySelector('#avatar-stillness').checked;

            const referenceFile = form.querySelector('#avatar-reference').files[0];
            if (referenceFile) {
                const reader = new FileReader();
                reader.onload = () => {
                    this.referenceImage = reader.result.split(',')[1];
                };
                reader.readAsDataURL(referenceFile);
            }

            modal.remove();
        };

        modal.appendChild(form);
        document.body.appendChild(modal);
    }

    // Load available voices
    async loadVoices() {
        try {
            const response = await fetch('http://localhost:8000/voices');
            return await response.json();
        } catch (error) {
            console.error('Error loading voices:', error);
            return [];
        }
    }

    // Load available styles
    async loadStyles() {
        try {
            const response = await fetch('http://localhost:8000/styles');
            return await response.json();
        } catch (error) {
            console.error('Error loading styles:', error);
            return [];
        }
    }

    // Add extension to SillyTavern
    addToSillyTavern() {
        // Add avatar toggle to chat controls
        const chatControls = document.querySelector('.chat-controls');
        if (chatControls) {
            const avatarToggle = document.createElement('button');
            avatarToggle.textContent = '👤';
            avatarToggle.style.cssText = `
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                padding: 5px;
            `;
            avatarToggle.onclick = () => {
                this.avatarContainer.style.display = 
                    this.avatarContainer.style.display === 'none' ? 'block' : 'none';
            };
            chatControls.appendChild(avatarToggle);
        }

        // Override send message to include avatar
        const originalSendMessage = window.sendMessage;
        window.sendMessage = async (message) => {
            // Call original function
            await originalSendMessage(message);

            // Send to avatar
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({
                    type: 'text',
                    text: message,
                    voice: this.voice,
                    model: this.model,
                    temperature: this.temperature,
                    speed: this.speed,
                    animation_style: this.animationStyle,
                    motion_scale: this.motionScale,
                    stillness: this.stillness,
                    reference_image: this.referenceImage
                }));
            }
        };
    }
}

// Helper function to convert base64 to ArrayBuffer
function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}

// Initialize extension when SillyTavern is ready
window.addEventListener('load', () => {
    const extension = new SentientAvatarExtension();
    extension.init();
}); 