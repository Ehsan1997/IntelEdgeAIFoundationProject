import argparse
import cv2
from inference import Network
import numpy as np

INPUT_STREAM = "test_video.mp4"
# For Linux - Use the following
# CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"
# For windows - specifically on my system, please do update for linux.
CPU_EXTENSION = r'C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll'

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    t_desc = "The probability threshold for the classifier"
    c_desc = "The color of the bounding box"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-t", help=t_desc, default=0.55)
    optional.add_argument("-c", help=c_desc, default="g")
    args = parser.parse_args()

    return args


def infer_on_video(args):
    ### TODO: Initialize the Inference Engine
    inference_network = Network()
    ### TODO: Load the network model into the IE
    inference_network.load_model(args.m, args.d, CPU_EXTENSION)
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Get the color of the bounding box
    colors = {'r': (0, 0, 255), 'g': (0, 255, 0), 'b': (255,0,0)}
    color = colors[args.c]

    # Image to paste (small_image)
    ## Use opacity to paste the smaller image
    s_im_org = cv2.imread("p1.png", -1)

    
    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        frame_input = preprocessing(frame, 300, 300)

        ### TODO: Perform inference on the frame
        inference_network.async_inference(frame_input)
        status = inference_network.wait()

        ### TODO: Get the output of inference
        output = inference_network.extract_output()
        
        ### TODO: Update the frame to include detected bounding boxes
        
        # Resize the frame
        frame_input = reshape_output(frame_input, 300, 300).transpose(1,2,0)
        
        for box in output[0][0]:
            if box[2] > float(args.t):
                points = box[3:]
                points[[0,2]] = points[[0,2]] * width
                points[[1,3]] = points[[1,3]] * height
                points = points.astype(int)
                print("Drawing a bounding box")
                cv2.rectangle(frame, (points[0], points[1]), (points[2], points[3]), color, 2)
                # Drawing a line, on the corner of the bounding box
                ## Final point for line
                f_point = ((points[0] - 10), (points[1] - 10))
                cv2.line(frame, (points[0], points[1]), f_point, color, 2)
                # Pasting the small image over the frame
                
                # Resize the image according to bounding box
                s_im_w, s_im_h = abs(points[2]-points[0])//2, abs(points[1]-points[3])//2
                s_im = cv2.resize(s_im_org, (s_im_w, s_im_h), interpolation = cv2.INTER_AREA)
                # Use opacity to determine the dominant areas
                alpha_s = s_im[:,:, 3] / 255
                alpha_l = 1.0 - alpha_s

                h_offset, w_offset, _ = s_im.shape
                h_offset, w_offset = f_point[1] - h_offset, f_point[0] - w_offset
                if frame[h_offset:f_point[1], w_offset:f_point[0], :].shape == (s_im_h, s_im_w, 3):
                    # Add the channel values, so that the lower values become -
                    # transparent (smaller image)
                    for c in range(0,3):
                        frame[h_offset:f_point[1], w_offset:f_point[0], c] = (alpha_s * s_im[:,:,c] +
                                                                                alpha_l * 
                                                                                frame[h_offset:f_point[1], w_offset:f_point[0], c])
                

        # This is the edit that put's the custom image next to the bounding box
        
        # Write out the frame
        out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
def preprocessing(input_image, height, width):
    '''
    Given an input image, height and width:
    - Resize to width and height
    - Transpose the final "channel" dimension to be first
    - Reshape the image to add a "batch" of 1 at the start 
    '''
    image = np.copy(input_image)
    image = cv2.resize(image, (width, height))
    image = image.transpose((2,0,1))
    image = image.reshape(1, 3, height, width)

    return image

def reshape_output(output, height, width):
    """
    Change the height and width according to the params.
    """
    b, c, h, w = output.shape
    resized_output = np.copy(output)
    resized_output = resized_output.reshape(c,h,w).transpose(1, 2, 0)
    resized_output = cv2.resize(resized_output, (width, height))
    resized_output = resized_output.transpose(2, 0, 1)
    
    return resized_output


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
