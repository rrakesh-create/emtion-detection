
import os
import shutil
import random
import soundfile as sf
import librosa
import numpy as np
from gtts import gTTS
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from tqdm import tqdm

# Configuration
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_AUDIO_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio', 'telugu')
TARGET_AUDIO_DIR = os.path.join(project_dir, 'datasets', 'telugu_audio_synthetic', 'telugu')

# Ensure target directory exists
if not os.path.exists(TARGET_AUDIO_DIR):
    os.makedirs(TARGET_AUDIO_DIR)

# Emotions mappping
EMOTIONS = ['angry', 'happy', 'nuetral', 'sad', 'suprised'] # Note: 'nuetral' and 'suprised' match directory names

# Text for TTS (Telugu Sentences for each emotion)
# Using a small set of example sentences. In a real scenario, this list would be much larger.
TTS_SENTENCES = {
    'angry': [
        "నాకు చాలా కోపంగా ఉంది!", "నీవు చేసింది తప్పు.", "ఇది సహించరానిది.", "నన్ను విసిగించకు!", "ఇక్కడి నుండి వెళ్లిపో.",
        "నోరు మూసుకో!", "నీకు ఎంత ధైర్యం?", "నేను నిన్ను క్షమించను.", "ఇది అన్యాయం!", "నా కళ్ళ ముందు నుండి పో.",
        "మళ్ళీ ఇలా చేయవద్దు.", "నాకు చిరాకు తెప్పించకు.", "నీ ప్రవర్తన నాకు నచ్చలేదు.", "ఎందుకు ఇలా చేశావు?", "నేను చాలా సీరియస్ గా ఉన్నాను.",
        "ఇది మూర్ఖత్వం.", "నాతో మాట్లాడకు.", "అంతా నాశనం చేశావు.", "నేను నిన్ను నమ్మను.", "ఇది చెత్త పని."
    ],
    'happy': [
        "నాకు చాలా సంతోషంగా ఉంది.", "ఈ రోజు చాలా బాగుంది.", "శుభవార్త!", "మనం గెలిచాం!", "అభినందనలు.",
        "ఇది అద్భుతమైన రోజు.", "నాకు చాలా ఆనందంగా ఉంది.", "ఎంత మంచి వార్త చెప్పారు!", "నేను చాలా ఉత్సాహంగా ఉన్నాను.", "అంతా మంచికే జరిగింది.",
        "ఇది నా జీవితంలో అత్యుత్తమ రోజు.", "నవ్వుతూ ఉండండి.", "పార్టీ చేసుకుందాం!", "నా కల నిజమైంది.", "మీరు చాలా బాగా చేశారు.",
        "సంతోషం అంటే ఇదేనేమో.", "చాలా థాంక్స్!", "ఆహా, ఎంత బాగుందో!", "నేను గాలిలో తేలుతున్నాను.", "అంతా శుభమే."
    ],
    'nuetral': [
        "ఇది ఒక పుస్తకం.", "సమయం ఎంత?", "నేను ఇంటికి వెళ్తున్నాను.", "వర్షం పడుతోంది.", "నా పేరు రవి.",
        "ఇది ఒక పెన్ను.", "బస్సు ఎప్పుడు వస్తుంది?", "తలుపు తీయండి.", "నాకు మంచినీళ్లు కావాలి.", "రేపు కలుద్దాం.",
        "అతను బడికి వెళ్ళాడు.", "ఆకాశం నీలంగా ఉంది.", "ఇది నా ఇల్లు.", "అన్నం తిన్నావా?", "నాకు పని ఉంది.",
        "టీవీ ఆన్ చేయ్.", "ఇప్పుడు సమయం ఐదు గంటలు.", "ఆమె పాట పాడుతోంది.", "నేను పుస్తకం చదువుతున్నాను.", "కారు రోడ్డు మీద ఉంది."
    ],
    'sad': [
        "నాకు చాలా బాధగా ఉంది.", "నేను ఒంటరిగా ఉన్నాను.", "అంతా అయిపోయింది.", "ఎందుకు ఇలా జరిగింది?", "నాకు ఏడుపు వస్తోంది.",
        "నా గుండె పగిలిపోయింది.", "ఆశలన్నీ అడుగంటాయి.", "ఎవరూ నన్ను అర్థం చేసుకోవడం లేదు.", "జీవితం చాలా కష్టంగా ఉంది.", "నేను ఓడిపోయాను.",
        "నాకు ఎవరూ లేరు.", "ఇది చాలా దురదృష్టకరం.", "నా స్నేహితుడు దూరమయ్యాడు.", "మనసు బాగోలేదు.", "కన్నీళ్లు ఆగడం లేదు.",
        "ఎందుకు దేవుడా ఈ శిక్ష?", "నాకు బతకాలని లేదు.", "అంతా చీకటిగా ఉంది.", "నేను చాలా అలసిపోయాను.", "నా తప్పు ఏంటి?"
    ],
    'suprised': [
        "నిజమా? నేను నమ్మలేకపోతున్నాను!", "అద్భుతం!", "ఇది ఎలా సాధ్యం?", "వాయ్! ఇది చాలా పెద్దది.", "నువ్వు వచ్చావా?",
        "ఇంత తొందరగానా?", "ఇది నిజంగా జరిగిందా?", "ఓరి దేవుడా!", "నేను షాక్ అయ్యాను.", "అది ఏమిటి?",
        "అలా ఎలా జరిగింది?", "నమ్మశక్యంగా లేదు.", "నువ్వా ఇక్కడ?", "ఎంత ఆశ్చర్యం!", "అరే, ఇది ఎప్పుడు కొన్నావు?",
        "ఇది కలతో కూడా ఊహించలేదు.", "ఇంత అందంగా ఉందా!", "వావ్, సూపర్!", "ఊహించని పరిణామం.", "నిజంగానా?"
    ]
}

# Augmentation Pipeline
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

def generate_tts_data():
    print("Generating TTS Data...")
    for emotion, sentences in TTS_SENTENCES.items():
        print(f"Processing emotion: {emotion}, Sentences: {len(sentences)}")
        emotion_dir = os.path.join(TARGET_AUDIO_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        
        for i, text in enumerate(sentences):
            try:
                # print(f"Generating TTS {i+1}/{len(sentences)} for {emotion}...") 
                tts = gTTS(text=text, lang='te', slow=False)
                filename = f"tts_{emotion}_{i+1}.mp3"
                filepath = os.path.join(emotion_dir, filename)
                tts.save(filepath)
                
                data, sr = librosa.load(filepath, sr=None)
                wav_filename = filename.replace('.mp3', '.wav')
                wav_filepath = os.path.join(emotion_dir, wav_filename)
                sf.write(wav_filepath, data, sr)
                os.remove(filepath)
            except Exception as e:
                print(f"Error generating TTS for {emotion} index {i}: {e}")

def augment_existing_data():
    print("Augmenting Existing Data...")
    
    # Check if source exists
    if not os.path.exists(SOURCE_AUDIO_DIR):
        print(f"Source directory not found: {SOURCE_AUDIO_DIR}")
        return

    # Collect all audio files first
    all_files = []
    for root, dirs, files in os.walk(SOURCE_AUDIO_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                all_files.append(os.path.join(root, file))
    
    print(f"Found {len(all_files)} files to augment.")

    # Process files
    for source_path in tqdm(all_files, desc="Augmenting"):
        try:
            # Determine relative path to maintain structure
            # root is dirname of source_path
            root = os.path.dirname(source_path)
            rel_path = os.path.relpath(root, SOURCE_AUDIO_DIR)
            target_subdir = os.path.join(TARGET_AUDIO_DIR, rel_path)
            os.makedirs(target_subdir, exist_ok=True)
            
            # Load audio
            data, sr = librosa.load(source_path, sr=None)
            
            # Apply augmentation 5 times per file (Increased for better DL performance)
            for i in range(5):
                augmented_data = augmenter(samples=data, sample_rate=sr)
                
                # Construct new filename
                file = os.path.basename(source_path)
                base_name, ext = os.path.splitext(file)
                new_filename = f"{base_name}_aug_{i+1}{ext}" 
                if not new_filename.lower().endswith('.wav'):
                        new_filename = f"{base_name}_aug_{i+1}.wav"
                
                target_path = os.path.join(target_subdir, new_filename)
                
                # Save
                sf.write(target_path, augmented_data, sr)
                
        except Exception as e:
            print(f"Error augmenting {source_path}: {e}")

if __name__ == "__main__":
    print("Starting Synthetic Data Generation...")
    
    try:
        generate_tts_data()
        augment_existing_data()
        print("\nSynthetic Data Generation Completed Successfully!")
        print(f"Data saved to: {TARGET_AUDIO_DIR}")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
