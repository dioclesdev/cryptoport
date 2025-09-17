#!/usr/bin/env python3
"""
Bereinigt persönliche Pfade in config.py
"""

import os
import re
from pathlib import Path

# Arbeite im Verzeichnis wo das Script liegt
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_PATH.parent
os.chdir(PROJECT_DIR)

def clean_config_paths():
    """Ersetzt absolute Pfade durch relative/generische Pfade"""
    config_path = PROJECT_DIR / 'utils' / 'config.py'
    
    if not config_path.exists():
        print("❌ utils/config.py nicht gefunden")
        return False
    
    print(f"🔧 Bereinige Pfade in {config_path}...")
    
    # Backup
    backup_path = config_path.with_suffix('.py.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"💾 Backup: {backup_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Original problematische Zeilen
    old_cache_dirs = """    CACHE_DIRS = [
        '/home/slimbook/Dokumente/PythonDev/crypto_cache',
        '/home/slimbook/Dokumente/PythonDev/novumLabs/13_Crypto Analyzer/analysis',
        'local_cache'
    ]"""
    
    # Neue generische Pfade
    new_cache_dirs = """    CACHE_DIRS = [
        os.path.join(os.path.expanduser('~'), '.cryptoport', 'cache'),
        os.path.join(os.path.dirname(__file__), '..', 'cache'),
        'cache'
    ]"""
    
    # Ersetzen
    content = content.replace(old_cache_dirs, new_cache_dirs)
    
    # Auch die doppelten os.environ.get() bereinigen
    content = re.sub(
        r"os\.environ\.get\('(\w+)', os\.environ\.get\('\1', ''\)\)",
        r"os.environ.get('\1', '')",
        content
    )
    
    # Sicherstellen dass os importiert ist
    if 'import os' not in content:
        lines = content.split('\n')
        # Nach den Docstrings einfügen
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith('#') and not line.startswith('"""'):
                lines.insert(i, 'import os')
                lines.insert(i+1, 'from dotenv import load_dotenv')
                lines.insert(i+2, '')
                lines.insert(i+3, 'load_dotenv()')
                lines.insert(i+4, '')
                break
        content = '\n'.join(lines)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Pfade bereinigt - keine persönlichen Verzeichnisse mehr!")
    return True

def verify_no_personal_paths():
    """Überprüft ob noch persönliche Pfade vorhanden sind"""
    print("\n🔍 Suche nach persönlichen Pfaden...")
    
    personal_patterns = [
        '/home/slimbook',
        'novumLabs',
        '13_Crypto Analyzer'
    ]
    
    found_issues = False
    
    # Suche in allen Python-Dateien
    for py_file in PROJECT_DIR.rglob('*.py'):
        if 'backup' in str(py_file):
            continue
            
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for pattern in personal_patterns:
            if pattern in content:
                print(f"⚠️  {py_file.relative_to(PROJECT_DIR)}: enthält '{pattern}'")
                found_issues = True
    
    if not found_issues:
        print("✅ Keine persönlichen Pfade gefunden!")
    
    return not found_issues

def main():
    print("🚀 CryptoPort Path Cleaner")
    print("="*40)
    
    # Bereinige config.py
    if clean_config_paths():
        # Verifiziere
        verify_no_personal_paths()
        
        print("\n📋 Nächste Schritte:")
        print("1. Prüfen Sie utils/config.py")
        print("2. Testen Sie: python app.py")
        print("3. Git commit durchführen")
    else:
        print("\n❌ Bereinigung fehlgeschlagen")

if __name__ == "__main__":
    main()
