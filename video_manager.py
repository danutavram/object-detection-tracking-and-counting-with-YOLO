# video_manager.py

import cv2

def open_video(video_path):
    """
    Deschide un fisier video pentru a putea fi procesat.
    Args:
        video_path (str): Calea catre fisierul video de intrare.
    Returns:
        cap (cv2.VideoCapture): Obiectul care permite citirea video-ului.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Eroare la deschiderea videoclipului.")
        return None
    return cap

def initialize_video_writer(frame_width=1920, frame_height=1080, output_path="output_video.mp4", fps=60):
    """
    Initializeaza VideoWriter pentru a salva videoclipul procesat.
    Args:
        frame_width (int): Latimea cadrului video.
        frame_height (int): Inaltimea cadrului video.
        output_path (str): Calea catre fisierul de iesire.
        fps (int): Numarul de cadre pe secunda.
    Returns:
        out (cv2.VideoWriter): Obiectul care permite scrierea video-ului procesat.
    """
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    return out

def release_video(cap, out):
    """
    Elibereaza resursele video-ului.
    Args:
        cap (cv2.VideoCapture): Obiectul video care trebuie eliberat.
        out (cv2.VideoWriter): Obiectul VideoWriter care trebuie eliberat.
    """
    cap.release()
    out.release()
    cv2.destroyAllWindows()
