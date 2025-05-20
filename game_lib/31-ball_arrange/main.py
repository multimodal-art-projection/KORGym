# main.py
from text_ui import main_text
from image_ui import main_image_ui

def main():
    # Global configuration
    config = {
        "seed": 42,
        "level": 1,           # e.g. level=1 => 3 colors, 3+2 tubes
        "history_mode": False # True => store multiple images; False => overwrite one
    }
    print("Select mode:")
    print("1) Pure Text UI")
    print("2) Text + Image UI")
    mode = input("Enter 1 or 2: ")
    
    if mode == "1":
        main_text(config)
    elif mode == "2":
        main_image_ui(config)
    else:
        print("Invalid selection. Exiting.")

if __name__ == "__main__":
    main()
