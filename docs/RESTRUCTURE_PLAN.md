# MERS Project Restructuring & Migration Plan

## 1. Project Architecture (Implemented)
The project has been restructured to follow industry-standard Python/Flutter project layout.

```
mers_project/
├── assets/                 # Static assets
│   ├── models/             # .pth, .pkl files (Visual/Audio models)
│   ├── metrics/            # Session metrics CSVs
│   └── logs/               # Log files
├── config/                 # Configuration
│   └── settings.py         # Centralized configuration (Paths, Constants)
├── docs/                   # Documentation
│   ├── AUDIT_REPORT.md
│   ├── BUILD_APK.md
│   ├── RESTRUCTURE_PLAN.md
│   └── ...
├── evaluation_results/     # Model evaluation reports/matrices
├── mobile/                 # Flutter Application
│   ├── android/            # Android native code
│   ├── lib/                # Dart source code
│   │   ├── main.dart
│   │   ├── home_screen.dart
│   │   └── sync_service.dart # WebSocket Sync & Caching
│   └── pubspec.yaml
├── scripts/                # Utility and Training scripts
│   ├── audit/              # Compliance/Audit scripts
│   ├── benchmark/          # Performance benchmarking
│   ├── client/             # PC Client scripts
│   ├── evaluation/         # Model evaluation scripts
│   ├── pipeline/           # Automation pipelines
│   ├── setup/              # Environment setup/check
│   └── training/           # Training scripts (Audio/Visual)
├── src/                    # Core Python Source Code
│   └── mers/
│       ├── __init__.py
│       ├── api/            # FastAPI Server
│       │   └── server.py   # WebSocket/HTTP Endpoints
│       ├── core/           # Core Engines
│       │   ├── audio_engine.py
│       │   ├── visual_engine.py (Rule-based)
│       │   ├── visual_engine_cnn.py (Deep Learning)
│       │   ├── fusion_engine.py
│       │   └── ...
│       ├── ui/             # Desktop UI
│       │   └── dashboard.py
│       └── utils/          # Shared utilities
├── tests/                  # Unit and Integration tests
├── requirements.txt        # Python Dependencies
└── main.py                 # Unified Entry Point (CLI/GUI/Server)
```

## 2. Migration Status
**Phase 1: Preparation** [COMPLETED]
- Directory structure created.
- Backup of legacy code in `old_mers_backup/`.

**Phase 2: Relocation** [COMPLETED]
- Core engines moved to `src/mers/core/`.
- Models moved to `assets/models/`.
- Server logic moved to `src/mers/api/server.py`.
- Desktop UI moved to `src/mers/ui/dashboard.py`.
- Scripts organized into `scripts/` subdirectories.
- Documentation consolidated in `docs/`.

**Phase 3: Code Refactoring** [COMPLETED]
- Imports updated to absolute paths (e.g., `from mers.core...`).
- `config.settings` used for dynamic paths.
- `main.py` updated to support `--mode ui`, `--mode server`, and `--no-camera`.

**Phase 4: Mobile Implementation** [COMPLETED]
- Flutter project structure established in `mobile/`.
- `SyncService` implemented with WebSocket support and Hive caching.
- Android permissions (Camera/Mic) configured.
- UI enhancements: Responsive layout, Confidence Graph, Emotion Breakdown.
- APK build configuration fixed (Record dependency).

## 3. Mobile-PC Sync Architecture
- **Protocol**: WebSocket (FastAPI) for real-time bidirectional data.
- **Visual Channel**: Mobile sends frames -> Server processes -> Server sends predictions -> Mobile displays.
- **Audio Channel**: Mobile records -> Server processes -> Server sends predictions.
- **Offline Mode**: Mobile caches data using Hive; syncs when connection restored.
- **Configuration**: Dynamic Server IP configuration via Mobile UI.

## 4. Verification & Testing
- **Server**: Run `python main.py --mode server`.
- **Desktop UI**: Run `python main.py --mode ui`.
- **Mobile App**:
    - Install APK.
    - Connect to Server IP (displayed on Server start).
    - Verify "Connected" status.
    - Test Camera/Mic inputs.
    - Verify Emotion Graph updates.
