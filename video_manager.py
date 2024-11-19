# video_manager.py
import cv2

def open_video(video_path):
    """
    Deschide un fișier video pentru a putea fi procesat.
    Args:
        video_path (str): Calea către fișierul video de intrare.
    Returns:
        cap (cv2.VideoCapture): Obiectul care permite citirea video-ului.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Eroare la deschiderea videoclipului.")
        return None
    return cap

def initialize_video_writer(frame_width, frame_height, output_path, fps=240):
    """
    Initializează VideoWriter pentru a salva videoclipul procesat.
    Args:
        frame_width (int): Lățimea cadrului video.
        frame_height (int): Înălțimea cadrului video.
        output_path (str): Calea către fișierul de ieșire.
        fps (int): Numărul de cadre pe secundă.
    Returns:
        out (cv2.VideoWriter): Obiectul care permite scrierea video-ului procesat.
    """
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    return out

def release_video(cap, out):
    """
    Eliberează resursele video-ului.
    Args:
        cap (cv2.VideoCapture): Obiectul video care trebuie eliberat.
        out (cv2.VideoWriter): Obiectul VideoWriter care trebuie eliberat.
    """
    cap.release()
    out.release()
    cv2.destroyAllWindows()
