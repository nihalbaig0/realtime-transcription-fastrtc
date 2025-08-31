// Wait for the DOM to be fully loaded before running any code
document.addEventListener('DOMContentLoaded', function() {
    // Global variables for WebRTC connection
    let peerConnection;
    let webrtc_id;
    let audioContext, analyser, audioSource;
    let audioLevel = 0;
    let animationFrame;
    let isRecording = false;
    let eventSource;

    // DOM element references
    const startButton = document.getElementById('start-button');
    const transcriptDiv = document.getElementById('transcript');

    // Streaming transcript management
    let currentStreamingText = '';
    let lastCompleteText = '';
    let streamingSpan = null;
    let isStreaming = false;

    console.log('DOM loaded with streaming support');

    function showError(message) {
        console.error('Error:', message);
        const toast = document.getElementById('error-toast');
        toast.textContent = message;
        toast.style.display = 'block';

        setTimeout(() => {
            toast.style.display = 'none';
        }, 5000);
    }

    function handleMessage(event) {
        const eventJson = JSON.parse(event.data);
        if (eventJson.type === "error") {
            showError(eventJson.message);
        }
        console.log('Received message:', event.data);
    }

    function updateButtonState() {
        if (peerConnection && (peerConnection.connectionState === 'connecting' || peerConnection.connectionState === 'new')) {
            startButton.innerHTML = `
                <div class="icon-with-spinner">
                    <div class="spinner"></div>
                    <span>Connecting...</span>
                </div>
            `;
            isRecording = false;
        } else if (peerConnection && peerConnection.connectionState === 'connected') {
            startButton.innerHTML = `
                <div class="pulse-container">
                    <div class="pulse-circle streaming-pulse"></div>
                    <span>Streaming Live</span>
                </div>
            `;
            isRecording = true;
            isStreaming = true;
        } else {
            startButton.innerHTML = 'Start Streaming';
            isRecording = false;
            isStreaming = false;
        }
    }

    function setupAudioVisualization(stream) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        } else {
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }
        
        analyser = audioContext.createAnalyser();
        audioSource = audioContext.createMediaStreamSource(stream);
        audioSource.connect(analyser);
        analyser.fftSize = 64;
        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        function updateAudioLevel() {
            analyser.getByteFrequencyData(dataArray);
            const average = Array.from(dataArray).reduce((a, b) => a + b, 0) / dataArray.length;
            audioLevel = average / 255;

            // Update pulse circle with streaming indication
            const pulseCircle = document.querySelector('.pulse-circle');
            if (pulseCircle) {
                const streamingIntensity = 1 + (audioLevel * 0.5); // More subtle scaling
                pulseCircle.style.setProperty('--audio-level', streamingIntensity);
                
                // Add streaming class for continuous animation
                if (isStreaming) {
                    pulseCircle.classList.add('streaming-pulse');
                }
            }

            animationFrame = requestAnimationFrame(updateAudioLevel);
        }
        updateAudioLevel();
    }

    // Enhanced transcript management for streaming
    function appendStreamingTranscript(text, isPartial = false) {
        const cleanText = text.trim();
        if (!cleanText) return;

        if (isPartial) {
            // Handle partial/streaming updates
            handleStreamingUpdate(cleanText);
        } else {
            // Handle complete sentences
            handleCompleteUpdate(cleanText);
        }

        // Auto-scroll to bottom
        requestAnimationFrame(() => {
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
        });
    }

    function handleStreamingUpdate(newText) {
        // Create or update streaming span
        if (!streamingSpan) {
            streamingSpan = document.createElement('span');
            streamingSpan.className = 'streaming-text';
            
            // Create new paragraph if needed
            let currentP = transcriptDiv.querySelector('p.current-streaming');
            if (!currentP) {
                currentP = document.createElement('p');
                currentP.className = 'current-streaming';
                transcriptDiv.appendChild(currentP);
            }
            currentP.appendChild(streamingSpan);
        }

        // Update streaming text with intelligent merging
        const mergedText = intelligentTextMerge(currentStreamingText, newText);
        currentStreamingText = mergedText;
        streamingSpan.textContent = mergedText;
        
        // Add visual streaming indicator
        streamingSpan.classList.add('typing');
        
        // Remove typing indicator after brief delay
        setTimeout(() => {
            if (streamingSpan) {
                streamingSpan.classList.remove('typing');
            }
        }, 200);
    }

    function handleCompleteUpdate(finalText) {
        // Finalize streaming text
        if (streamingSpan) {
            const currentP = streamingSpan.parentElement;
            if (currentP) {
                currentP.className = 'completed';
                currentP.textContent = finalText;
            }
            streamingSpan = null;
        }
        
        currentStreamingText = '';
        lastCompleteText = finalText;
    }

    function intelligentTextMerge(current, incoming) {
        if (!current) return incoming;
        
        // Remove common prefixes to avoid duplication
        const words = current.split(' ');
        const incomingWords = incoming.split(' ');
        
        // Find common suffix in current and prefix in incoming
        let overlapIndex = -1;
        const minLength = Math.min(words.length, incomingWords.length);
        
        for (let i = 1; i <= minLength; i++) {
            const currentSuffix = words.slice(-i).join(' ').toLowerCase();
            const incomingPrefix = incomingWords.slice(0, i).join(' ').toLowerCase();
            
            if (currentSuffix === incomingPrefix) {
                overlapIndex = i;
            }
        }
        
        if (overlapIndex > 0) {
            // Merge by removing overlap
            const newPart = incomingWords.slice(overlapIndex).join(' ');
            return newPart ? current + ' ' + newPart : current;
        }
        
        // No overlap found, simple concatenation
        return current + ' ' + incoming;
    }

    async function setupWebRTC() {
        console.log('Setting up WebRTC for streaming transcription...');
        
        try {
            const config = window.__RTC_CONFIGURATION__;
            peerConnection = new RTCPeerConnection(config);

            const connectionTimeout = setTimeout(() => {
                if (peerConnection && peerConnection.connectionState !== 'connected') {
                    showError('Connection timeout. Please check your network and try again.');
                    stop();
                }
            }, 15000);

            updateButtonState();

            console.log('Requesting microphone for streaming...');
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,  // Optimize for speech recognition
                    autoGainControl: true,
                    noiseSuppression: true,
                    echoCancellation: true
                }
            });

            setupAudioVisualization(stream);

            stream.getTracks().forEach(track => {
                peerConnection.addTrack(track, stream);
            });

            peerConnection.addEventListener('connectionstatechange', () => {
                console.log('Connection state:', peerConnection.connectionState);
                
                if (peerConnection.connectionState === 'connected') {
                    clearTimeout(connectionTimeout);
                    const toast = document.getElementById('error-toast');
                    toast.style.display = 'none';
                    console.log('Streaming connection established');
                } else if (peerConnection.connectionState === 'failed' || 
                          peerConnection.connectionState === 'disconnected' || 
                          peerConnection.connectionState === 'closed') {
                    showError('Streaming connection lost. Please try again.');
                    stop();
                }
                updateButtonState();
            });

            const dataChannel = peerConnection.createDataChannel('text');
            dataChannel.onmessage = handleMessage;

            peerConnection.onicecandidate = ({ candidate }) => {
                if (candidate) {
                    fetch('/webrtc/offer', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            candidate: candidate.toJSON(),
                            webrtc_id: webrtc_id,
                            type: "ice-candidate",
                        })
                    });
                }
            };

            const offer = await peerConnection.createOffer();
            await peerConnection.setLocalDescription(offer);

            webrtc_id = Math.random().toString(36).substring(7);
            console.log('Generated webrtc_id for streaming:', webrtc_id);

            const response = await fetch('/webrtc/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sdp: peerConnection.localDescription.sdp,
                    type: peerConnection.localDescription.type,
                    webrtc_id: webrtc_id
                })
            });

            const serverResponse = await response.json();

            if (serverResponse.status === 'failed') {
                showError(serverResponse.meta.error === 'concurrency_limit_reached'
                    ? `Too many connections. Maximum limit is ${serverResponse.meta.limit}`
                    : serverResponse.meta.error);
                stop();
                return;
            }

            await peerConnection.setRemoteDescription(serverResponse);

            // Set up streaming transcript reception
            console.log('Setting up streaming transcript connection...');
            eventSource = new EventSource('/transcript?webrtc_id=' + webrtc_id);
            
            eventSource.onerror = (event) => {
                console.error("Streaming EventSource error:", event);
                showError("Streaming connection lost. Please try again.");
            };
            
            // Handle streaming events
            eventSource.addEventListener("streaming", (event) => {
                console.log("Received streaming chunk:", event.data);
                appendStreamingTranscript(event.data, true);  // Mark as partial
            });
            
            // Handle final/complete events if implemented
            eventSource.addEventListener("complete", (event) => {
                console.log("Received complete transcript:", event.data);
                appendStreamingTranscript(event.data, false);  // Mark as complete
            });
            
            console.log('Streaming WebRTC setup complete');
        } catch (err) {
            console.error('Error setting up streaming WebRTC:', err);
            showError('Failed to establish streaming connection: ' + err.message);
            stop();
        }
    }

    function stop() {
        console.log('Stopping streaming transcription...');
        
        if (animationFrame) {
            cancelAnimationFrame(animationFrame);
            animationFrame = null;
        }
        
        if (audioContext) {
            audioContext.suspend();
        }
        
        if (peerConnection) {
            const senders = peerConnection.getSenders();
            if (senders) {
                senders.forEach(sender => {
                    if (sender.track) {
                        sender.track.stop();
                    }
                });
            }
            peerConnection.close();
            peerConnection = null;
        }
        
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }

        // Clean up server-side resources
        if (webrtc_id) {
            fetch(`/cleanup/${webrtc_id}`, { method: 'POST' }).catch(console.error);
        }
        
        // Finalize any streaming text
        if (streamingSpan && currentStreamingText) {
            handleCompleteUpdate(currentStreamingText);
        }
        
        audioLevel = 0;
        isStreaming = false;
        updateButtonState();
        
        if (window.confirm('Clear streaming transcript?')) {
            transcriptDiv.innerHTML = '';
            currentStreamingText = '';
            lastCompleteText = '';
            streamingSpan = null;
        } else {
            // Finalize current streaming paragraph
            const currentP = transcriptDiv.querySelector('p.current-streaming');
            if (currentP) {
                currentP.className = 'completed';
            }
        }
    }

    window.addEventListener('beforeunload', () => {
        stop();
    });

    startButton.addEventListener('click', () => {
        console.log('Streaming button clicked. isRecording:', isRecording);
        if (!isRecording) {
            setupWebRTC();
        } else {
            stop();
        }
    });

    // Add CSS for streaming effects
    const style = document.createElement('style');
    style.textContent = `
        .streaming-text {
            background: linear-gradient(90deg, transparent, rgba(249, 164, 92, 0.3), transparent);
            background-size: 200% 100%;
            border-radius: 3px;
            padding: 2px 4px;
            margin: 0 1px;
        }
        
        .streaming-text.typing {
            animation: streamingGlow 1s ease-in-out;
        }
        
        .streaming-pulse {
            animation: streamingPulse 1s ease-in-out infinite !important;
        }
        
        .current-streaming {
            border-left: 3px solid rgba(249, 164, 92, 0.8);
            padding-left: 8px;
            margin-left: 4px;
            background: rgba(249, 164, 92, 0.05);
        }
        
        .completed {
            border-left: 3px solid rgba(249, 164, 92, 0.3);
            padding-left: 8px;
            margin-left: 4px;
        }
        
        @keyframes streamingGlow {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        
        @keyframes streamingPulse {
            0% { opacity: 0.6; transform: scale(0.9); }
            50% { opacity: 1; transform: scale(1.1); }
            100% { opacity: 0.6; transform: scale(0.9); }
        }
    `;
    document.head.appendChild(style);

    console.log('Streaming client initialization complete');
});
