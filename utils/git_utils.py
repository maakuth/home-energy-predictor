import subprocess

def get_git_version():
    """Get the current git head (short hash)."""
    try:
        version = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            stderr=subprocess.STDOUT
        ).decode('ascii').strip()
        return version
    except Exception as e:
        print(f"Warning: Could not get git version: {e}")
        return "unknown"
