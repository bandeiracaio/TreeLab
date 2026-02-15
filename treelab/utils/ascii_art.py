"""ASCII art and text decorations for TreeLab (no emojis)."""


def get_treelab_banner():
    """Get ASCII art banner for TreeLab."""
    return """
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ████████╗██████╗ ███████╗███████╗██╗      █████╗ ██████╗    ║
║  ╚══██╔══╝██╔══██╗██╔════╝██╔════╝██║     ██╔══██╗██╔══██╗   ║
║     ██║   ██████╔╝█████╗  █████╗  ██║     ███████║██████╔╝   ║
║     ██║   ██╔══██╗██╔══╝  ██╔══╝  ██║     ██╔══██║██╔══██╗   ║
║     ██║   ██║  ██║███████╗███████╗███████╗██║  ██║██████╔╝   ║
║     ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝╚═════╝    ║
║                                                                ║
║                      v0.3.0                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""


def get_simple_banner():
    """Get simple ASCII banner."""
    return """
  _____              _          _     
 |_   _| __ ___  ___| |    __ _| |__  
  | || '__/ _ \\/ _ \\ |   / _` | '_ \\ 
  | || | |  __/  __/ |__| (_| | |_) |
  |_||_|  \\___|\\___|_____\\__,_|_.__/ 
                                        
                    v0.3.0
"""


def get_mode_indicator(mode):
    """Get ASCII art for mode indicator."""
    if mode.lower() == "transformation":
        return """
┌─────────────────────────┐
│   MODE: TRANSFORMATION  │
│   [Data Preprocessing]  │
└─────────────────────────┘
"""
    else:
        return """
┌─────────────────────────┐
│     MODE: MODELING      │
│   [Model Training]      │
└─────────────────────────┘
"""


def get_success_icon():
    """Success indicator."""
    return "[OK]"


def get_error_icon():
    """Error indicator."""
    return "[ERROR]"


def get_warning_icon():
    """Warning indicator."""
    return "[WARN]"


def get_info_icon():
    """Info indicator."""
    return "[INFO]"


def get_checkpoint_icon():
    """Checkpoint indicator."""
    return "[*]"


def get_action_icon():
    """Action indicator."""
    return "[>]"
