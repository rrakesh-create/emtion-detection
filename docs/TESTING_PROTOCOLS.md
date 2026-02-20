# MERS Testing Protocols

This document outlines the testing procedures for verifying the MERS (Multimodal Emotion Recognition System) functionality, specifically focusing on the new Mobile-PC Synchronization and Restructured Architecture.

## 1. System Setup Verification

### 1.1 Directory Structure Check
**Objective:** Ensure the project follows the new structure.
- [ ] Verify `src/mers/core` contains `visual_engine.py`, `audio_engine.py`.
- [ ] Verify `src/mers/api` contains `server.py`.
- [ ] Verify `assets/models` contains `.pth` and `.task` files.
- [ ] Verify `config/settings.py` is the single source of truth for paths.

### 1.2 Entry Point Test
**Objective:** Ensure `main.py` correctly launches components.
- [ ] Run `python main.py --help` -> Should show `--mode` options.
- [ ] Run `python main.py --mode ui` -> Should launch Desktop Dashboard.
- [ ] Run `python main.py --mode server` -> Should launch FastAPI server on port 8000.

---

## 2. Mobile-PC Synchronization Testing

**Prerequisites:**
- PC running MERS Server (`python main.py --mode server`).
- Mobile device/emulator on the SAME network as PC.
- Mobile App installed/running.

### 2.1 Connection & Handshake
1.  **Start Server**: Note the IP address printed in terminal (e.g., `192.168.1.X`).
2.  **Launch App**: Open MERS Mobile Client.
3.  **Configure IP**: Enter `http://<PC_IP>:8000` in the Server IP field and press Save/Enter.
4.  **Verify**:
    - Status card should turn **Green**.
    - Text should say "Connected to Server".
    - Toast/SnackBar confirmation "Server IP Updated".

### 2.2 Real-Time Visual Sync
1.  **Setup**: Ensure PC Server has a webcam active (or is processing video).
2.  **Action**: Make an emotional face (e.g., Happy) at the PC camera.
3.  **Verify Mobile**:
    - The "Live Visual Status" section on mobile should update in near real-time (latency < 500ms).
    - Text should reflect the emotion detected by PC (e.g., "Visual: Happy").

### 2.3 Audio Analysis (Bidirectional)
1.  **Action**: Press and hold "Hold to Record" on mobile.
2.  **Speak**: Say a sentence with emotion (e.g., "I am very angry right now!").
3.  **Release**: Release the button to send audio.
4.  **Verify**:
    - Status text changes: Recording -> Analyzing -> Analysis Complete.
    - "Audio Analysis Result" section appears with Emotion and Confidence.
    - PC Server logs should show "Received audio file" and "Analysis result sent".

---

## 3. Mobile Offline & Caching Capabilities

### 3.1 Network Disconnection
1.  **Setup**: Establish connection (Green status).
2.  **Action**: Turn off Wi-Fi on Mobile Device.
3.  **Verify**:
    - Status card turns **Red** ("Disconnected").
    - **Crucial**: The last known "Live Visual Status" should REMAIN visible (persisted via Hive), not disappear.

### 3.2 Automatic Reconnection
1.  **Action**: Turn Wi-Fi back ON.
2.  **Wait**: Wait up to 5-10 seconds.
3.  **Verify**:
    - Status card automatically turns **Green** without user intervention.
    - Real-time updates resume.

### 3.3 App Restart Persistence
1.  **Action**: Kill the mobile app completely.
2.  **Action**: Relaunch app.
3.  **Verify**:
    - The "Live Visual Status" should populate with the *last cached value* immediately, even before connection is established.

---

## 4. UI/UX & Orientation Testing

### 4.1 Portrait Mode
- [ ] Hold phone vertically.
- [ ] Verify single-column layout.
- [ ] Verify all elements (Status, IP, Button, Results) are stacked and scrollable.

### 4.2 Landscape Mode
- [ ] Rotate phone horizontally.
- [ ] Verify **Split-Screen Layout**:
    - **Left Side**: Status, IP Config, Live Visual Data.
    - **Right Side**: Huge Recording Button, Audio Results.
- [ ] Verify scrolling works independently on both sides if content overflows.

---

## 5. Troubleshooting Common Issues

- **Connection Refused**: Check Windows Firewall (Allow Python/FastAPI on port 8000).
- **Audio Error**: Ensure `ffmpeg` is installed on Server for audio conversion.
- **Permission Denied**: Ensure Mobile App has Microphone permissions granted.
