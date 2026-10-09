import torch
import platform
import psutil

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
            
        # Device
        device = ResourceChecker.get_available_device("auto")
        print(f"Recommended Compute Device: {device}")
        print("===============================\n")

if __name__ == "__main__":
    ResourceChecker.print_resource_report()
