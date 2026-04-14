#!/usr/bin/env python3
"""
Generate a report on character voices available in the characters directory.
Lists the narrated voice, file count, average file length, and total length for each character.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

def get_audio_duration(filepath: str) -> float:
    """
    Get the duration of an audio file in seconds using ffprobe.
    
    Args:
        filepath: Path to the audio file
        
    Returns:
        Duration in seconds, or 0 if unable to determine
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
             filepath],
            capture_output=True,
            text=True,
            timeout=10
        )
        return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return 0.0

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    if seconds == 0:
        return "0s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)

def analyze_character(char_dir: Path) -> Dict:
    """
    Analyze a character directory and return statistics.
    
    Args:
        char_dir: Path to the character directory
        
    Returns:
        Dictionary with character analysis data
    """
    char_name = char_dir.name
    
    # Extract voice type from directory name (e.g., "Kaladin_male" -> "male")
    parts = char_name.rsplit('_', 1)
    voice_type = parts[-1] if len(parts) > 1 else "unknown"
    
    # Get all .wav files
    wav_files = sorted(list(char_dir.glob("*.wav")))
    
    if not wav_files:
        return {
            "name": char_name,
            "voice_type": voice_type,
            "file_count": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "files": []
        }
    
    # Get durations for all files
    durations = []
    for wav_file in wav_files:
        duration = get_audio_duration(str(wav_file))
        durations.append(duration)
    
    total_duration = sum(durations)
    average_duration = total_duration / len(durations) if durations else 0.0
    
    return {
        "name": char_name,
        "voice_type": voice_type,
        "file_count": len(wav_files),
        "total_duration": total_duration,
        "average_duration": average_duration,
        "files": [f.name for f in wav_files]
    }

def generate_report(characters_dir: Path) -> None:
    """
    Generate and display a report of all character voices.
    
    Args:
        characters_dir: Path to the characters directory
    """
    # Find all character subdirectories
    char_dirs = sorted([d for d in characters_dir.iterdir() if d.is_dir()])
    
    if not char_dirs:
        print("No character directories found.")
        return
    
    # Analyze each character
    results = []
    for char_dir in char_dirs:
        analysis = analyze_character(char_dir)
        results.append(analysis)
    
    # Display report
    print("\n" + "=" * 100)
    print("CHARACTER VOICES REPORT")
    print("=" * 100)
    print()
    
    for result in results:
        print(f"Character: {result['name']}")
        print(f"  Voice Type:        {result['voice_type']}")
        print(f"  File Count:        {result['file_count']}")
        
        if result['file_count'] > 0:
            print(f"  Average Duration:  {format_duration(result['average_duration'])}")
            print(f"  Total Duration:    {format_duration(result['total_duration'])}")
        else:
            print(f"  Average Duration:  No files")
            print(f"  Total Duration:    No files")
        
        print()
    
    # Summary statistics
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    total_characters = len(results)
    total_files = sum(r['file_count'] for r in results)
    total_duration = sum(r['total_duration'] for r in results)
    
    print(f"Total Characters:   {total_characters}")
    print(f"Total Files:        {total_files}")
    print(f"Total Duration:     {format_duration(total_duration)}")
    print()

if __name__ == "__main__":
    # Get the path to the characters directory
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    characters_dir = project_root / "characters"
    
    if not characters_dir.exists():
        print(f"Error: Characters directory not found at {characters_dir}")
        exit(1)
    
    generate_report(characters_dir)
