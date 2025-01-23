
import cv2 
import os

images_folder = 'C:/Users/ksiri/OneDrive/Desktop/Resized images' # Adjust this path as needed 
output_folder = 'C:/Users/ksiri/OneDrive/Desktop/detected images' 

os.makedirs(output_folder, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for root, dirs, files in os.walk(images_folder):
    files.sort()
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {filename}.") 
                continue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            relative_path = os.path.relpath(root, images_folder)
            output_subfolder = os.path.join(output_folder, relative_path)
            os.makedirs(output_subfolder, exist_ok=True)
            output_filename = f"detected_{filename}"
            output_path = os.path.join(output_subfolder, output_filename)
            cv2.imwrite(output_path, image)
