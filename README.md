# MPICV
### Python application utilizing MPI to accelerate OpenCV compute tasks

The goal of this project was to demonstrate feasability of using MPI as a means of distributing compute among multiple nodes for OpenCV workloads.

This program can perform movement detection based on frame mask delta, as well as perform face detection using MTCNN module with pre-trained deep neural network. Gaussian blur can then be applied to the detected face to introduced further CPU load.

There are multiple options for an input to the program. The default (no argument) is internal webcam of the laptop program is running on. Alternatively, video can be supplied using `--video` argument or an RTSP network stream using `--rtsp` argument.

By default, program performs movement detection and can be switched to face detection using `--face` argument. To enable face blur, `--blur` argument must be supplied.

### Installation
- Clone this project
- Create and enable venv
`python3 -m venv venv`
`source venv/bin/activate`
- Install requirements
`pip install -r requirements.txt`

### Running the program
`python3 mpicv.py`
` 

