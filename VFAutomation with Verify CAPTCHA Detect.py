import cv2
import numpy as np
import pyautogui
import time
import keyboard

# Set the path to the button image
button_path = r'C:\Users\eanet\OneDrive\Documents\VFAutomation_farm_button_screenshot.png'
stop_image_path = r'C:\Users\eanet\OneDrive\Documents\VFAutomation_stop_image_screenshot.png'

# Load the button image as a NumPy array
button_image = cv2.imread(button_path)
stop_image = cv2.imread(stop_image_path)

# Ensure the button image is not None
if button_image is not None and stop_image is not None:
    stop_loop = False

    def on_key_press(event):
        global stop_loop
        if event.name == 'q':
            stop_loop = True

    # Register the key press event handler
    keyboard.on_press(on_key_press)

    while not stop_loop:
        # Capture the screen
        screenshot = pyautogui.screenshot()
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Perform image recognition for the button
        result_button = cv2.matchTemplate(screenshot, button_image, cv2.TM_CCOEFF_NORMED)
        threshold_button = 0.8

        # Perform image recognition for the stop image
        result_stop = cv2.matchTemplate(screenshot, stop_image, cv2.TM_CCOEFF_NORMED)
        threshold_stop = 0.8

        # Find the coordinates of matches above the threshold
        locations_button = np.where(result_button >= threshold_button)

        if locations_button[0].size > 0:
            # Get the coordinates of the best match
            best_match_button = np.unravel_index(result_button.argmax(), result_button.shape)
            button_x, button_y = best_match_button[::-1]

            # Calculate the center of the button
            button_width, button_height = button_image.shape[1], button_image.shape[0]
            center_x = button_x + button_width // 2
            center_y = button_y + button_height // 2

            # Perform a click action at the center of the button
            pyautogui.click(center_x, center_y)

        # Find the coordinates of matches above the threshold for the stop image
        locations_stop = np.where(result_stop >= threshold_stop)

        if locations_stop[0].size > 0:
            print("Stop image detected. Stopping the script.")
            break

        # Pause for 3 seconds
        time.sleep(3)

        # Check if 'q' key is pressed to stop the script
        if stop_loop:
            print("Stop signal received. Stopping the script.")
            break

    cv2.destroyAllWindows()
else:
    print("Failed to load the button or stop image. Please check the file paths.")