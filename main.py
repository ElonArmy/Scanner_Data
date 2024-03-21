# main.py
from image_processor import ImageProcessor
from interface import load_and_process_image

if __name__ == "__main__":
    relative_path = load_and_process_image()
    ImageProcessor(relative_path).run()