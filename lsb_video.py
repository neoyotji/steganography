#LSB algoritması

"""It can choose a cover image of size M*N as an input.

The message to be hidden is embedded in RGB element only of an image.

It can need a pixel selection filter to acquire the best location to hide information in the cover image to acquire a better cost.

The filter can be used to Least Significant Bit (LSB) of each pixel to conceal record, leaving most significant bits (MSB).

After that Message is hidden utilizing Bit Replacement method."""

import cv2
import numpy as np

def steganography(video_path: str, message: str, frame_indices: list) -> None:
    """
    Performs steganography on a video using the least significant bit (LSB) method.

    Parameters:
    - video_path: str
        The path to the video file.
    - message: str
        The message to be hidden in the video.
    - frame_indices: list
        A list of frame indices where the message will be inserted.

    Returns:
    - None

    Raises:
    - FileNotFoundError:
        If the video file is not found.
    """

    # Load the video
    try:
        video = cv2.VideoCapture("C:\Users\SenanurOzbag\Downloads\video.mp4")
    except FileNotFoundError:
        raise FileNotFoundError("Video file not found.")

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert the message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)

    # Check if the number of frames is greater than the number of frame indices provided
    if total_frames < len(frame_indices):
        raise ValueError("Number of frame indices exceeds the total number of frames in the video.")

    # Iterate over the frame indices
    for index in frame_indices:
        # Check if the frame index is valid
        if index < 0 or index >= total_frames:
            raise ValueError(f"Invalid frame index: {index}")

        # Read the frame
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = video.read()

        # Check if the frame is read successfully
        if not ret:
            raise ValueError(f"Failed to read frame at index: {index}")

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the height and width of the frame
        height, width = gray_frame.shape

        # Get the length of the binary message
        message_length = len(binary_message)

        # Check if the message can fit in the frame
        if message_length > height * width:
            raise ValueError("Message is too long to fit in the frame.")

        # Flatten the frame into a 1D array
        flat_frame = gray_frame.flatten()

        # Modify the least significant bit of each pixel in the frame
        for i in range(message_length):
            # Get the binary representation of the pixel value
            binary_pixel = format(flat_frame[i], '08b')

            # Modify the least significant bit of the pixel value
            modified_pixel = binary_pixel[:-1] + binary_message[i]

            # Convert the modified pixel value back to decimal
            modified_pixel_decimal = int(modified_pixel, 2)

            # Update the pixel value in the frame
            flat_frame[i] = modified_pixel_decimal

        # Reshape the modified frame back to its original shape
        modified_frame = flat_frame.reshape(height, width)

        # Save the modified frame to a new video file
        output_video_path = f"steganography_output_{index}.avi"
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        output_video.write(cv2.cvtColor(modified_frame, cv2.COLOR_GRAY2BGR))
        output_video.release()

    # Release the video capture
    video.release()

# Example usage:
video_path = "input_video.avi"
message = "This is a secret message"
frame_indices = [10, 20, 30]
steganography(video_path, message, frame_indices)