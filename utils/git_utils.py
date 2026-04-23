import subprocess
import os

def get_model_version():
    """
    Get the semantic version of the model training code.
    
    This version is manually updated in VERSION file when model training
    logic changes. It's NOT tied to git commits - only intentional model
    changes increment this version.
    
    See AGENTS.md: "Model Versioning" for when to update VERSION.
    """
    try:
        version_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION')
        with open(version_file, 'r') as f:
            version = f.read().strip()
        if version:
            return version
    except Exception as e:
        print(f"Warning: Could not read VERSION file: {e}")
    return "unknown"


def get_git_version():
    """
    DEPRECATED: Get the current git head (short hash).
    
    Use get_model_version() instead for model tracking.
    This is kept for backward compatibility only.
    """
    try:
        version = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], 
            stderr=subprocess.STDOUT
        ).decode('ascii').strip()
        return version
    except Exception as e:
        print(f"Warning: Could not get git version: {e}")
        return "unknown"
