# Building MERS Mobile APK

This guide outlines the steps to build the MERS mobile application APK for Android.

## Prerequisites

1.  **Flutter SDK**: Ensure Flutter is installed and in your PATH.
2.  **Android SDK**:
    *   Install Command Line Tools.
    *   Install Platform Tools and Build Tools (tested with build-tools;34.0.0).
    *   Accept Android Licenses: `flutter doctor --android-licenses`.
3.  **Java JDK**: Ensure Java is installed (JDK 17 recommended).

## Project Setup

1.  Navigate to the `mobile` directory:
    ```bash
    cd mobile
    ```

2.  **Important Dependency Note**:
    Due to version mismatches in the `record` package's Linux dependency, you must add a dependency override in `pubspec.yaml` if you encounter `RecordLinux` errors.
    
    Ensure `pubspec.yaml` contains:
    ```yaml
    dependencies:
      record: 5.0.0
    
    dependency_overrides:
      record_platform_interface: 1.0.1
    ```

3.  Install dependencies:
    ```bash
    flutter pub get
    ```

## Building the APK

Run the following command to build the release APK:

```bash
flutter build apk --release
```

## Output

The built APK will be located at:
`build/app/outputs/flutter-apk/app-release.apk`

## Installing on Device

1.  Connect your Android device via USB.
2.  Enable **USB Debugging** on your device.
3.  Run:
    ```bash
    flutter install
    ```
    Or transfer the APK file manually to your device and install it.

## Troubleshooting

*   **License Errors**: Run `flutter doctor --android-licenses` and accept all licenses.
*   **Symlink Errors (Windows)**: Run `flutter config --no-enable-windows-desktop` (and other unused platforms) to avoid plugin symlink issues if Developer Mode is not enabled.
*   **Build Tools Missing**: Use `sdkmanager "build-tools;34.0.0"` to install.
