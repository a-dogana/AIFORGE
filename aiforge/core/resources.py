import torch
import platform
import psutil
import os
try:
    import resource
except ImportError:
    resource = None

class ResourceChecker:
    @staticmethod
    def get_available_device(requested_device: str = "auto") -> str:
        """
        Determines the best available device based on the request.
        """
        if requested_device == "cpu":
            return "cpu"
            
        if requested_device in ("auto", "mps"):
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                    
        if requested_device in ("auto", "cuda"):
            if torch.cuda.is_available():
                return "cuda"
                
        return "cpu"

    @staticmethod
    def check_and_raise_nofile_limit(target_limit: int = 10240) -> dict:
        """
        Checks the open file descriptor limit (RLIMIT_NOFILE).
        On macOS/Linux, attempts to proactively raise the soft limit up to target_limit or hard limit.
        """
        if resource is None:
            return {"supported": False, "soft": -1, "hard": -1, "warning": None}
            
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            raised = False
            
            # If soft limit is too low (common on macOS GUI apps where soft=256), try to increase it
            if soft < target_limit:
                new_soft = min(target_limit, hard) if hard > 0 else target_limit
                if new_soft > soft:
                    try:
                        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
                        soft = new_soft
                        raised = True
                    except (ValueError, OSError):
                        pass

            warning = None
            if soft < 1024:
                warning = f"Open file limit is low ({soft}). Run 'ulimit -n 10240' to prevent system socket exhaustion on macOS."
                
            return {
                "supported": True,
                "soft": soft,
                "hard": hard,
                "raised": raised,
                "warning": warning
            }
        except Exception as e:
            return {"supported": False, "error": str(e), "soft": -1, "hard": -1, "warning": None}

    @staticmethod
    def print_resource_report():
        """
        Prints a report of available system resources.
        """
        print("=== AIFORGE Resource Report ===")
        print(f"Platform: {platform.system()} {platform.machine()} {platform.release()}")
        
        # RAM
        ram_info = psutil.virtual_memory()
        total_ram_gb = ram_info.total / (1024 ** 3)
        available_ram_gb = ram_info.available / (1024 ** 3)
        print(f"Total RAM: {total_ram_gb:.2f} GB")
        print(f"Available RAM: {available_ram_gb:.2f} GB")
        
        if total_ram_gb < 8:
            print("WARNING: Total RAM is less than 8GB. Large experiments may fail.")
            
        # File descriptor guard
        fd_guard = ResourceChecker.check_and_raise_nofile_limit()
        if fd_guard.get("supported"):
            print(f"File Descriptor Limit (Soft/Hard): {fd_guard['soft']}/{fd_guard['hard']}")
            if fd_guard.get("warning"):
                print(f"WARNING: {fd_guard['warning']}")

        # Device
        device = ResourceChecker.get_available_device("auto")
        print(f"Recommended Compute Device: {device}")
        print("===============================\n")

if __name__ == "__main__":
    ResourceChecker.print_resource_report()
