# React Native Migration Guide
## From HTML/JS to Mobile App with FastAPI Backend

---

## Overview

You'll migrate your voice translation app from browser (HTML/JS) to a native mobile app using React Native while keeping your FastAPI backend unchanged.

**Architecture:**
```
React Native App (Mobile)
        ↓ WebSocket
FastAPI Backend (Same as before)
        ↓
  ASR → MT → TTS Pipeline
```

---

## Phase 1: Setup React Native Project

### 1. Install Prerequisites

```bash
# Install Node.js (if not already installed)
# Download from: https://nodejs.org/

# Install React Native CLI
npm install -g react-native-cli

# For Android: Install Android Studio
# For iOS: Install Xcode (Mac only)
```

### 2. Create New Project

```bash
# Create React Native project
npx react-native init VoiceTranslationApp

cd VoiceTranslationApp

# Install required packages
npm install @react-native-community/netinfo
npm install react-native-webrtc
npm install @react-native-async-storage/async-storage
```

---

## Phase 2: Audio Capture Setup

### Install Audio Libraries

```bash
# For microphone access and audio recording
npm install react-native-audio-recorder-player

# For real-time audio streaming
npm install react-native-live-audio-stream
```

### Configure Permissions

**Android: `android/app/src/main/AndroidManifest.xml`**
```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    <!-- Add these permissions -->
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" />
    
    <application>
        ...
    </application>
</manifest>
```

**iOS: `ios/VoiceTranslationApp/Info.plist`**
```xml
<key>NSMicrophoneUsageDescription</key>
<string>We need microphone access for voice translation</string>
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

---

## Phase 3: Core Components

### 1. WebSocket Service (`src/services/WebSocketService.js`)

```javascript
class WebSocketService {
  constructor() {
    this.ws = null;
    this.listeners = {};
  }

  connect(roomId, userId, nativeLanguage, onMessage, onError) {
    const wsUrl = `ws://YOUR_BACKEND_IP:8000/ws/call/${roomId}/${userId}`;
    
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      // Send initial config
      this.ws.send(JSON.stringify({
        native_lang: nativeLanguage
      }));
      
      if (this.listeners.onConnect) {
        this.listeners.onConnect();
      }
    };
    
    this.ws.onmessage = (event) => {
      if (typeof event.data === 'string') {
        // JSON message
        const msg = JSON.parse(event.data);
        onMessage(msg);
      } else {
        // Binary audio data
        onMessage({ type: 'audio', data: event.data });
      }
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      onError(error);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket disconnected');
      if (this.listeners.onDisconnect) {
        this.listeners.onDisconnect();
      }
    };
  }

  sendAudio(audioData) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  on(event, callback) {
    this.listeners[event] = callback;
  }
}

export default new WebSocketService();
```

### 2. Audio Streaming Service (`src/services/AudioStreamService.js`)

```javascript
import LiveAudioStream from 'react-native-live-audio-stream';

class AudioStreamService {
  constructor() {
    this.isStreaming = false;
  }

  async start(onAudioData) {
    const options = {
      sampleRate: 16000,  // Match your backend
      channels: 1,         // Mono
      bitsPerSample: 16,   // 16-bit PCM
      audioSource: 6,      // VOICE_RECOGNITION for Android
      bufferSize: 8192     // Chunk size
    };

    LiveAudioStream.init(options);
    
    LiveAudioStream.on('data', data => {
      // data is base64 encoded PCM audio
      if (onAudioData) {
        // Convert base64 to binary
        const audioBuffer = Buffer.from(data, 'base64');
        onAudioData(audioBuffer);
      }
    });

    LiveAudioStream.start();
    this.isStreaming = true;
    console.log('Audio streaming started');
  }

  stop() {
    if (this.isStreaming) {
      LiveAudioStream.stop();
      this.isStreaming = false;
      console.log('Audio streaming stopped');
    }
  }
}

export default new AudioStreamService();
```

### 3. Audio Playback Service (`src/services/AudioPlaybackService.js`)

```javascript
import AudioRecorderPlayer from 'react-native-audio-recorder-player';
import RNFS from 'react-native-fs';

class AudioPlaybackService {
  constructor() {
    this.audioRecorderPlayer = new AudioRecorderPlayer();
  }

  async playAudioFromBlob(audioBlob) {
    try {
      // Save blob to temp file
      const tempPath = `${RNFS.CachesDirectoryPath}/temp_audio.wav`;
      
      // Convert blob to base64
      const base64Audio = await this.blobToBase64(audioBlob);
      
      // Write to file
      await RNFS.writeFile(tempPath, base64Audio, 'base64');
      
      // Play audio
      await this.audioRecorderPlayer.startPlayer(tempPath);
      
      this.audioRecorderPlayer.addPlayBackListener((e) => {
        if (e.currentPosition === e.duration) {
          this.audioRecorderPlayer.stopPlayer();
          // Clean up temp file
          RNFS.unlink(tempPath);
        }
      });
      
    } catch (error) {
      console.error('Playback error:', error);
    }
  }

  blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result.split(',')[1];
        resolve(base64);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  }

  async stop() {
    await this.audioRecorderPlayer.stopPlayer();
  }
}

export default new AudioPlaybackService();
```

---

## Phase 4: Main Screen Component

### `App.js`

```javascript
import React, { useState, useEffect } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  TextInput,
  Picker
} from 'react-native';

import WebSocketService from './src/services/WebSocketService';
import AudioStreamService from './src/services/AudioStreamService';
import AudioPlaybackService from './src/services/AudioPlaybackService';

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [roomId, setRoomId] = useState('room1');
  const [userId] = useState(`user_${Math.random().toString(36).substr(2, 9)}`);
  const [nativeLanguage, setNativeLanguage] = useState('en');
  const [translatedText, setTranslatedText] = useState('Waiting for translation...');
  const [originalText, setOriginalText] = useState('-');
  const [status, setStatus] = useState('Disconnected');

  useEffect(() => {
    // Setup WebSocket listeners
    WebSocketService.on('onConnect', () => {
      setIsConnected(true);
      setStatus('Connected');
    });

    WebSocketService.on('onDisconnect', () => {
      setIsConnected(false);
      setIsStreaming(false);
      setStatus('Disconnected');
      AudioStreamService.stop();
    });

    return () => {
      handleLeaveCall();
    };
  }, []);

  const handleJoinCall = async () => {
    try {
      setStatus('Connecting...');
      
      // Connect WebSocket
      WebSocketService.connect(
        roomId,
        userId,
        nativeLanguage,
        handleWebSocketMessage,
        (error) => {
          console.error('WebSocket error:', error);
          setStatus('Connection Error');
        }
      );

      // Start audio streaming
      await AudioStreamService.start((audioData) => {
        // Send audio chunks to backend
        WebSocketService.sendAudio(audioData);
      });

      setIsStreaming(true);
      
    } catch (error) {
      console.error('Failed to join call:', error);
      setStatus('Failed to connect');
    }
  };

  const handleLeaveCall = () => {
    AudioStreamService.stop();
    WebSocketService.disconnect();
    setIsConnected(false);
    setIsStreaming(false);
    setStatus('Disconnected');
  };

  const handleWebSocketMessage = async (message) => {
    if (message.type === 'audio') {
      // Play received audio
      await AudioPlaybackService.playAudioFromBlob(message.data);
    } else if (message.type === 'caption') {
      setTranslatedText(message.text);
      setOriginalText(message.original);
    } else if (message.info) {
      setStatus(message.info);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>🌐 Voice Translation</Text>
      </View>

      {!isConnected && (
        <View style={styles.settingsContainer}>
          <Text style={styles.label}>Room ID:</Text>
          <TextInput
            style={styles.input}
            value={roomId}
            onChangeText={setRoomId}
            placeholder="Enter room ID"
          />

          <Text style={styles.label}>My Language:</Text>
          <Picker
            selectedValue={nativeLanguage}
            style={styles.picker}
            onValueChange={(itemValue) => setNativeLanguage(itemValue)}
          >
            <Picker.Item label="English" value="en" />
            <Picker.Item label="French" value="fr" />
            <Picker.Item label="German" value="de" />
            <Picker.Item label="Spanish" value="es" />
          </Picker>
        </View>
      )}

      <View style={styles.statusContainer}>
        <View style={[
          styles.statusIndicator,
          { backgroundColor: isConnected ? '#4CAF50' : '#F44336' }
        ]} />
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {isStreaming && (
        <View style={styles.recordingIndicator}>
          <View style={styles.pulseCircle} />
          <Text style={styles.recordingText}>Microphone Active</Text>
        </View>
      )}

      <View style={styles.captionContainer}>
        <Text style={styles.captionLabel}>Incoming Translation:</Text>
        <Text style={styles.captionText}>{translatedText}</Text>
      </View>

      <View style={styles.captionContainer}>
        <Text style={styles.captionLabel}>Original:</Text>
        <Text style={styles.captionTextSmall}>{originalText}</Text>
      </View>

      <View style={styles.buttonContainer}>
        {!isConnected ? (
          <TouchableOpacity
            style={[styles.button, styles.joinButton]}
            onPress={handleJoinCall}
          >
            <Text style={styles.buttonText}>Join Call</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.button, styles.leaveButton]}
            onPress={handleLeaveCall}
          >
            <Text style={styles.buttonText}>Leave Call</Text>
          </TouchableOpacity>
        )}
      </View>

      <View style={styles.infoContainer}>
        <Text style={styles.infoText}>
          ℹ️ Just speak naturally - the app automatically detects when you finish speaking and translates to the other person.
        </Text>
      </View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  header: {
    padding: 20,
    backgroundColor: '#007bff',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
  settingsContainer: {
    padding: 20,
    backgroundColor: 'white',
    margin: 10,
    borderRadius: 10,
  },
  label: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: 10,
    marginBottom: 5,
  },
  input: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
    padding: 10,
    fontSize: 16,
  },
  picker: {
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 5,
  },
  statusContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 15,
    backgroundColor: 'white',
    margin: 10,
    borderRadius: 10,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 10,
  },
  statusText: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  recordingIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 10,
    backgroundColor: '#ffe6e6',
    margin: 10,
    borderRadius: 10,
  },
  pulseCircle: {
    width: 12,
    height: 12,
    borderRadius: 6,
    backgroundColor: '#F44336',
    marginRight: 10,
  },
  recordingText: {
    color: '#F44336',
    fontWeight: 'bold',
  },
  captionContainer: {
    padding: 15,
    backgroundColor: 'white',
    margin: 10,
    borderRadius: 10,
    borderLeftWidth: 4,
    borderLeftColor: '#007bff',
  },
  captionLabel: {
    fontSize: 12,
    color: '#666',
    marginBottom: 5,
  },
  captionText: {
    fontSize: 18,
    color: '#333',
  },
  captionTextSmall: {
    fontSize: 14,
    color: '#666',
  },
  buttonContainer: {
    padding: 20,
  },
  button: {
    padding: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  joinButton: {
    backgroundColor: '#007bff',
  },
  leaveButton: {
    backgroundColor: '#F44336',
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  infoContainer: {
    padding: 15,
    backgroundColor: '#e7f3ff',
    margin: 10,
    borderRadius: 10,
  },
  infoText: {
    fontSize: 14,
    color: '#333',
  },
});

export default App;
```

---

## Phase 5: Backend Changes

### Update Your Backend IP Address

In your FastAPI backend, make sure it's accessible from your phone:

**Option A: Same WiFi Network**
```bash
# Find your computer's local IP
# Windows: ipconfig
# Mac/Linux: ifconfig

# Run backend with:
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Option B: Ngrok Tunnel (for testing)**
```bash
# Install ngrok
# Download from: https://ngrok.com/

# Tunnel to your backend
ngrok http 8000

# Use the ngrok URL in your React Native app
# ws://YOUR_NGROK_URL/ws/call/...
```

### Update CORS (if needed)

Add to your `main.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Phase 6: Build and Run

### Android

```bash
# Connect Android device via USB with USB debugging enabled
# OR start Android emulator

# Run app
npx react-native run-android
```

### iOS (Mac only)

```bash
# Install pods
cd ios && pod install && cd ..

# Run app
npx react-native run-ios
```

---

## Common Issues and Solutions

### Issue 1: Audio Not Streaming

**Solution:** Check microphone permissions
```javascript
import { PermissionsAndroid, Platform } from 'react-native';

async function requestMicrophonePermission() {
  if (Platform.OS === 'android') {
    const granted = await PermissionsAndroid.request(
      PermissionsAndroid.PERMISSIONS.RECORD_AUDIO
    );
    return granted === PermissionsAndroid.RESULTS.GRANTED;
  }
  return true;
}

// Call before starting audio
await requestMicrophonePermission();
```

### Issue 2: WebSocket Not Connecting

**Solution:** Make sure backend IP is correct
```javascript
// Test backend accessibility
fetch('http://YOUR_BACKEND_IP:8000/health')
  .then(res => console.log('Backend reachable'))
  .catch(err => console.error('Cannot reach backend:', err));
```

### Issue 3: Audio Quality Issues

**Solution:** Adjust audio parameters
```javascript
const options = {
  sampleRate: 16000,   // Match backend
  channels: 1,
  bitsPerSample: 16,
  audioSource: 6,      // VOICE_RECOGNITION
  bufferSize: 4096     // Try different sizes: 2048, 4096, 8192
};
```

---

## Testing Checklist

- [ ] Microphone permission granted
- [ ] WebSocket connects to backend
- [ ] Audio streams continuously
- [ ] VAD detects speech/silence
- [ ] Translation received and played
- [ ] Captions display correctly
- [ ] Leave call cleans up resources
- [ ] Works on both WiFi and mobile data (if using ngrok)

---

## Production Deployment

### Backend Hosting

**Option 1: Cloud Server (AWS, DigitalOcean, etc.)**
```bash
# Deploy FastAPI to cloud
# Use Nginx + Uvicorn for production
# Enable SSL (wss:// for WebSocket)
```

**Option 2: Serverless (AWS Lambda + API Gateway)**
- More complex setup
- WebSocket support via API Gateway
- May have latency issues

### App Store Deployment

**Android (Google Play)**
```bash
# Generate signed APK
cd android
./gradlew assembleRelease
```

**iOS (App Store)**
```bash
# Build in Xcode
# Archive and submit to App Store Connect
```

---

## Next Steps

1. **Test with current HTML frontend first** - Make sure VAD works
2. **Set up React Native project** - Get basic UI working
3. **Implement audio streaming** - Test microphone → WebSocket
4. **Add playback** - Test receiving and playing audio
5. **Polish UI** - Add animations, better error handling
6. **Deploy** - Cloud backend + app stores

---

## Estimated Timeline

- **Week 1:** VAD in HTML/JS (verify it works)
- **Week 2:** React Native setup + basic UI
- **Week 3:** Audio streaming + WebSocket integration
- **Week 4:** Polish + testing + deployment

Good luck! 🚀
