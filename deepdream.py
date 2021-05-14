#!/usr/bin/env python

# DeepDream OpenSource-Art
# Much of the source code here is derived from 
#      https://github.com/saturnism/deepdream-docker
# which in turn is derived from 
#     https://github.com/google/deepdream
# and modified to integrate into the OpenSource Art Project

from io import StringIO, BytesIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import argparse
import shutil
import os
os.environ["GLOG_minloglevel"] = "2"
import caffe
import sys
import random
import tempfile
import time

from subprocess import Popen, PIPE
import errno
from random import randint

def find_model(models_dir):
    '''find_model will look through a list of folders in some parent models
       directory, and check the folder for the prototext files. If both
       are found for a randomly chosen model, it is returned.

       note:: This model is currently not in use to select, since the models
             would need testing. It is provided so we can implement this :)
    '''
    # Save everything to return to user
    print(models_dir)
    if 'models' in models_dir:
        while models_dir[-6:] != 'models':
            models_dir = os.path.dirname(models_dir)
    print(models_dir)
    
    models = os.listdir(models_dir)
    print('Found %s candidate models in %s' %(len(models), models_dir))

    for i, model in enumerate(models):
        print(str(i) + ' - ' + model)


    while True:
        choice = input("Enter the number of model (or press q to quit) :\n> ")

        if type(choice) == str and choice.lower() == 'q':
            sys.exit(0)
        try:
            if int(choice) in range(len(models)):
                break
        except Exception:
            print('Wrong input! ' + choice + " is invalid, please try again.")


    direct_path = '' if os.environ.get('DEEPDREAM') else '.'

    model_path = os.path.join(models_dir, models[int(choice)])
    print('Downloading... : ' + model_path)

    
    print(Popen('python ' + direct_path + '/download_model_binary.py '+ model_path, shell=True,
                           stdout=PIPE).stdout.read())
    print('Done!')
    print(os.listdir(model_path))
    model_name = '-'.join([md for md in os.listdir(model_path) if '.caffemodel' in md])
    print('\nModel Path : ' + model_path)
    print('Model Name : ' + model_name)

    show_layers(model_path, model_name)


def show_layers(model_path, model_name):
    net_fn   = os.path.join(model_path, 'deploy.prototxt')
    param_fn = os.path.join(model_path, model_name)
    
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))
    
    net = caffe.Classifier('tmp.prototxt', param_fn) 
    print("\n>>> LAYERS for Model : " + model_name)
    print()
    seq = list(net.blobs)
    n_cols = 3
    data = [seq[i:i+n_cols] for i in range(0, len(seq), n_cols)]

    col_width = max(len(word) for row in data for word in row) + 2  # padding
    for row in data:
        print("".join(word.ljust(col_width) for word in row))

    os.remove('tmp.prototxt')


# -- Argument Parsing
def get_parser():
    parser = argparse.ArgumentParser(description="DeepDream Video with Caffe")
    parser.add_argument(
        '--mode',
        type=int,
        required=False,
        help="""What action(s) to perform:\n- 0: (default) run all (create frames, dream and recreate the video)\n- 1: extract frames only\n- 2: run deepdream only (make sure frames are already where they should be)
        \n- 3: make the video from already existing processed frames\n- 4: download a new model\n- 5: show layers (requires --model-name and --model-path if different from default)""",
        default=0,
        choices=range(6),
        dest='mode',)

    parser.add_argument(
        '-i','--input',
        type=str,
        help='Input directory where extracted frames are stored',
        # required='--extract' not in sys.argv and '-e' not in sys.argv and (parser.parse_args().mode not in [0,1,2,3]),
        required=False,
        default='./data/input_frames',
        dest='input_dir')

    parser.add_argument(
        '-e','--extract',
        type=str,
        help='Path to video to process',
        required=False,
        # required=('--input' not in sys.argv and '-i' not in sys.argv) and (parser.parse_args().mode not in [0,1,2,3]),
        dest='extract',
        )

    parser.add_argument(
        '-o','--output',
        type=str,
        help='Output directory where processed frames are to be stored',
        default='./data/input_frames',
        dest='output_dir')

    parser.add_argument(
        '-it','--image-type',
        help='Specify whether frames will be jpg or png ',
        default='jpg',
        type=str,
        required=False,
        dest='image_type')

    parser.add_argument(
        '-p', '--model-path',
        type=str,
        dest='model_path',
        required=False,
        default='caffe/models/bvlc_googlenet/',
        help='Model directory to use')
    parser.add_argument(
        '-m', '--model-name',
        type=str,
        required=False,
        dest='model_name',
        default='bvlc_googlenet.caffemodel',
        help='Caffe Model name to use')

    parser.add_argument(
        '-oct','--octaves',
        type=int,
        required=False,
        help='Octaves. Default: 4',
        default=4,
        dest='octaves',)
    parser.add_argument(
        '-octs','--octavescale',
        type=float,
        required=False,
        help='Octave Scale. Default: 1.4',
        default=1.4,
        dest='octavescale',)

    parser.add_argument(
        '-itr','--iterations',
        type=int,
        required=False,
        help='Iterations. Default: 10',
        default=10,
        )
    parser.add_argument(
        '-j','--jitter',
        type=int,
        required=False,
        help='Jitter. Default: 32',
        default=32,
        dest='jitter',)

    parser.add_argument(
        '-s','--stepsize',
        type=float,
        required=False,
        help='Step Size. Default: 1.5',
        default=1.5,
        dest='step_size',)

    parser.add_argument(
        '-b','--blend',
        type=str,
        required=False,
        help='Blend Amount. Default: "0.5" (constant), or "loop" (0.5-1.0), or "random"',
        default='0.5',
        dest='blend')
    parser.add_argument(
        '-l','--layers',
        nargs="+",
        type=str,
        required=False,
        help='List of Layers to loop through. Default: "customloop" \
        - or choose ie "inception_4c/output inception_4d/output" for those layers',
        default='customloop',
        dest='layers',)
    parser.add_argument(
        '-v', '--verbose',
        type=int,
        required=False,
        help="verbosity [1-3]",
        default=2,
        dest='verbose',)
    parser.add_argument(
        '-gi', '--guide-image',
        required=False,
        type=str,
        help="path to guide image",
        default=None,
        dest='guide_image',)
    parser.add_argument(
        '-sf', '--start-frame',
        type=int,
        required=False,
        help="starting frame number",
        default=1,
        dest='start_frame',)
    parser.add_argument(
        '-ef', '--end-frame',
        type=int,
        required=False,
        help="end frame number",
        default=None,
        dest='end_frame',)
    
    


    return parser

# -- Environment Variables

def get_envar(name, default=None):
    '''get an environment variable with "name" and
       check that it exists as a file or folder on the filesystem.
       If not defined, or doesn't exist, exit on error.
    '''
    value = os.environ.get(name, default)

    if not value:
        print('Please export %s' % value)
        sys.exit(1)

    if not os.path.exists(value):
        print('%s does not exist! Is it inside the container?' % value)
        sys.exit(1)

    return value
    

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


# -- DeepDream Functions

def objective_L2(dst):
    dst.diff[:] = dst.data 

def objective_guide(dst,guide_features):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

def make_step(net,
              step_size=1.5,
              end='inception_4c/output', 
              jitter=32,
              clip=True,
              objective=objective_L2):

    '''Basic gradient ascent step.'''

    # input image is stored in Net's 'data' blob
    src = net.blobs['data']
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]

    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)  


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, image_output=None, save_image=None, verbose=1, message=None, **step_params):
 
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]

    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    
    now = time.time()
    totaltime = 0
    
    for octave, octave_base in enumerate(octaves[::-1]):
    
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail

        for i in range(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            
            later = time.time()
            difference = int(later - now)
            totaltime += difference

            # save last image (this could be modified to save sequence or other)
            # if save_image is not None and image_output is not None: #and i==iter_n-1
            #     image_file = f"{image_output}/o{octave}_i{i}_{end.replace('/', '-')}-{difference}s{save_image}"

            #     # The keys have directory / in them, may need to mkdir
            #     image_dir = os.path.dirname(image_file)
            #     if not os.path.exists(image_dir):
            #         # not entirely safe way to do it, but ok for start_time
            #         os.makedirs(image_dir)

            #     PIL.Image.fromarray(np.uint8(vis)).save(image_file)

            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            # showarray(vis)
            if verbose > 1:
                # print(message)
                print("Octave:", octave, " - Iter:", i, " - Layer:", end, " - Shape:", vis.shape)
                clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])


# --------------
# Guided Dreaming
# --------------
def make_step_guided(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective_fn=objective_guide, **objective_params):
    '''Basic gradient ascent step.'''

    #if objective_fn is None:
    #    objective_fn = objective_L2

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective_fn(dst, **objective_params)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream_guided(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, verbose=1, objective_fn=objective_guide, **step_params):

    #if objective_fn is None:
    #    objective_fn = objective_L2

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in range(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details

    now = time.time()
    totaltime = 0

    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in range(iter_n):
            make_step_guided(net, end=end, clip=clip, objective_fn=objective_fn, **step_params)

            later = time.time()
            difference = int(later - now)
            totaltime += difference

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            # if verbose == 3:
            #     if image_type == "png":
            #         showarrayHQ(vis)
            #     elif image_type == "jpg":
            #         showarray(vis)
            #     print(octave, i, end, vis.shape)
            #     clear_output(wait=True)
            # if verbose == 2:
            #     print(octave, i, end, vis.shape)
            if verbose > 1:
                # print(message)
                print("Octave:", octave, " - Iter:", i, " - Layer:", end, " - Shape:", vis.shape)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])


    
    
def make_sure_path_exists(path):
    '''
    make sure input and output directory exist, if not create them.
    If another error (permission denied) throw an error.
    '''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

layersloop = ['inception_4c/output', 'inception_4d/output',
              'inception_4e/output', 'inception_5a/output',
              'inception_5b/output', 'inception_5a/output',
              'inception_4e/output', 'inception_4d/output',
              'inception_4c/output']

def prepare_guide(net, image, end="inception_4c/output", maxW=224, maxH=224):
        # grab dimensions of input image
        (w, h) = image.size

        # GoogLeNet was trained on images with maximum width and heights
        # of 224 pixels -- if either dimension is larger than 224 pixels,
        # then we'll need to do some resizing
        if h > maxH or w > maxW:
            # resize based on width
            if w > h:
                r = maxW / float(w)

            # resize based on height
            else:
                r = maxH / float(h)

            # resize the image
            (nW, nH) = (int(r * w), int(r * h))
            image = np.float32(image.resize((nW, nH), PIL.Image.BILINEAR))

        (src, dst) = (net.blobs["data"], net.blobs[end])
        src.reshape(1, 3, nH, nW)
        src.data[0] = preprocess(net, image)
        net.forward(end=end)
        guide_features = dst.data[0].copy()

        return guide_features



def main(input_dir, output_dir, image_type, model_path, model_name, octaves, octave_scale,
         iterations, jitter, step_size, blend, layers, guide_image, start_frame, end_frame, verbose):

    # make_sure_path_exists(input_dir)
    make_sure_path_exists(output_dir)

    #Load DNN
    net_fn   = os.path.join(model_path, 'deploy.prototxt')
    param_fn = os.path.join(model_path, model_name)

    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    
    # let max nr of frames
    nrframes =len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])

    if nrframes == 0:
        print("no frames to process found")
        sys.exit(0)
    
    # --- Dream!
    frame_start_time = time.time()
    totaltime = 0

    frame_i = 1 if start_frame is None else int(start_frame)

    nrframes = (nrframes+1) if end_frame is None else int(end_frame)+1
    
    if blend == 'loop':
        blend_forward = True
        blend_at = 0.4
        blend_step = 0.1

    try:
        blend = float(blend)
    except Exception:
        print("Blend: " + blend + "  " + type(blend))


    frame = np.float32(PIL.Image.open(input_dir + '/%08d.%s' % (frame_i, image_type) ))
    # image = input_dir + f"/{frame_i:08d}.{image_type}"
    # print(os.path.join(input_dir, f'/{frame_i:08d}.{image_type}'))
    # frame = np.float32(PIL.Image.open(image))


    print('nb frames: ', nrframes-1)

    for i in range(frame_i, nrframes):
        
        #Choosing Layer
        if layers == 'customloop': #loop over layers as set in layersloop array
            endparam = layersloop[(i-1) % len(layersloop)]
        else: #loop through layers one at a time until this specific layer
            endparam = layers[(i-1) % len(layers)]
            
            
        frame_start_time = time.time()
        print('START FRAME ' + str(i) + ' of ' + str(nrframes-1))

        if guide_image is None:

            frame = deepdream(net, frame, iter_n=iterations,octave_n=octaves, octave_scale=octave_scale, end=endparam, step_size = step_size, verbose=verbose, jitter=jitter,)
        else:
            print('Setting up Guide with selected image')
            guide_features = prepare_guide(net, PIL.Image.open(guide_image), end=endparam)

            frame = deepdream_guided(net, frame, verbose=verbose, iter_n = iterations, step_size = step_size, 
                                    octave_n = octaves, octave_scale = octave_scale, jitter=jitter, end=endparam, objective_fn=objective_guide, guide_features=guide_features,)

        later = time.time()
        difference = int(later - frame_start_time)    
        
        
        # saveframe = output_dir + f"/dream_frame_{i:08d}.{image_type}"
        saveframe = output_dir + "/dream_frame_%08d.%s" % (i, image_type)

        PIL.Image.fromarray(np.uint8(frame)).save(saveframe)
        print('DONE FRAME ' + str(i) + ' of ' + str(nrframes-1))
        
        totaltime += difference
        avgtime = (totaltime / (i))
        print('>> Frame Time: ' + str(int(later - frame_start_time)) + 's  ---  Total Time: ' +
              str(difference) + 's')
        timeleft = avgtime * ((nrframes-1) - i)        
        m, s = divmod(timeleft, 60)
        h, m = divmod(m, 60)
        print('>> Estimated Total Time Remaining: ' + str(timeleft) +
              's (' + "%d:%02d:%02d" % (h, m, s) + ')')


        if i == nrframes-1:
            break
        newframe = input_dir + "/%08d.%s" % (i, image_type) #f'/{(i+1):08d}.{image_type}' 
        if blend == 0:
            frame = np.float32(PIL.Image.open(newframe))
        else:
            if blend == 'random':
            	blendval=randint(5,10)/10.
            elif blend == 'loop':
                if blend_at > 1 - blend_step: blend_forward = False
                elif blend_at <= 0.5: blend_forward = True
                if blend_forward: blend_at += blend_step
                else: blend_at -= blend_step
                blendval = blend_at
            else: blendval = float(blend)

            frame = np.float32(morphPicture(saveframe, newframe, blendval))

        print('***************************************')
        
    print('DeepDreams are made of cheese, who am I to diss a brie?')

    # Remove temporary parameters, we could keep this if someone wanted
#     shutil.rmtree(tmpdir)


def morphPicture(filename1,filename2,blend):
	img1 = PIL.Image.open(filename1)
	img2 = PIL.Image.open(filename2)

	return PIL.Image.blend(img1, img2, blend)

def extract_video(video, ext, frame_dir):

    make_sure_path_exists(frame_dir)
    # print(Popen('ffmpeg -i ' + video + ' -f image2 ' + frame_dir + '/%08d.' + ext, shell=True,
    #                        stdout=PIPE).stdout.read())

    print('avconv -i ' + video + ' -f image2 ' + frame_dir + '/%08d.' + ext)
    print(Popen('avconv -i ' + video + ' -f image2 ' + frame_dir + '/%08d.' + ext, shell=True,
                           stdout=PIPE).stdout.read())

    print('Frames created to ' + frame_dir)

def create_video(frames_directory, original_video, ext, output_video,frame_rate=24):
    script_path = "/frames2movie.sh" if os.environ.get('DEEPDREAM_OUTPUT') else "./frames2movie.sh"

    # output = Popen((
    #     script_path + " ffmpeg " + frames_directory + " " + original_video + " " + ext + " " + output_video),
    #     shell=True, stdout=PIPE).stdout.read()

    output = Popen((
        script_path + " avconv " + frames_directory + " " + original_video + " " + ext + " " + output_video),
        shell=True, stdout=PIPE).stdout.read()
    print(script_path + " avconv " + frames_directory + " " + original_video + " " + ext + " " + output_video)
    print()


if __name__ == '__main__':

    parser = get_parser()

    def help(return_code=0):
        parser.print_help()
        sys.exit(return_code)
    
    # If the user didn't provide any arguments, show the full help
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    if args.mode in range(2):
        if not args.extract:
            help("MISSING ARGUMENT ERROR !\nArgument -e or --extract is required if --mode=0 or --mode=1")

    
    input_dir = os.environ.get('DEEPDREAM_INPUT', args.input_dir)
    output_dir = os.environ.get('DEEPDREAM_OUTPUT', args.output_dir)
    docker_path = os.environ.get('DEEPDREAM_MODELS')
    model_path = args.model_path

    if model_path == 'caffe/models/bvlc_googlenet/' and docker_path:
        model_path = model_path.replace('caffe', docker_path)

    if model_path[-1] == "/": model_path = model_path[:-1]
    if os.path.splitext(args.model_name)[0] != os.path.basename(model_path):
        model_path = model_path.replace(os.path.basename(model_path), os.path.splitext(args.model_name)[0])

    model = os.path.join(model_path, args.model_name)


    if not os.path.exists(model_path):
        print("Model directory not found : " + model_path)
        print("Please set the model_path to a correct caffe model directory")
        sys.exit(0)

    

    print("\n******************************************************************************")
    print("Model Path: " + model)

    if not os.path.exists(model):
        print("Model not found")
        print("Please set the model_name to a correct caffe model")
        print("or download one with for instance: ./caffe_dir/scripts/download_model_binary.py caffe_dir/models/bvlc_googlenet")
        sys.exit(0)


    if args.mode == 0:
        print('>>> MODE: FULL RUN')
    elif args.mode == 1:
        print('>>> MODE: Extract frames ONLY')
    elif args.mode == 2:
        print('>>> MODE: Process deepdream algorithm on frames ONLY')
    elif args.mode == 3:
        print('>>> MODE: Create Video from frames ONLY')

    if args.extract:
        video_name, _ = os.path.splitext(os.path.basename(args.extract))
        input_dir = os.path.join(input_dir, video_name + ('_guided' if args.guide_image else ''))
        output_dir = os.path.join(output_dir, video_name + ('_guided' if args.guide_image else ''))
        if args.mode == 0 or args.mode == 1: 
            print("******************************************************************************\n")
            print('\nExtracting frames..')
            extract_video(args.extract, args.image_type, input_dir)
            print("\n******************************************************************************")
    print("Input Directory : " + input_dir)
    print("Output Directory : " + output_dir)
    print("******************************************************************************\n")

    if args.mode == 0 or args.mode == 2: 
        print('\nStart dreaming..')

        main(input_dir=input_dir, 
            output_dir=output_dir, 
            image_type=args.image_type, 
            model_path=model_path, 
            model_name=args.model_name, 
            octaves=args.octaves, 
            octave_scale=args.octavescale, 
            iterations=args.iterations, 
            jitter=args.jitter,
            step_size=args.step_size, 
            blend=args.blend, 
            layers=args.layers, 
            guide_image=args.guide_image, 
            start_frame=args.start_frame, 
            end_frame=args.end_frame, 
            verbose=args.verbose)

    # recreate the video
    if args.mode == 0 or args.mode == 3:
        print('\nCreating video..')

        # if args.extract:
        t = str(time.time()).replace('.', '_')
        videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/dream_video")
        make_sure_path_exists(videos_dir)
        output_video = os.path.join(videos_dir, video_name + "-DeepDream" + ("_guided" if args.guide_image else '') + "__" + t)

        create_video(output_dir, args.extract, args.image_type, output_video)
        print(output_dir)

    if args.mode == 4:
        find_model(model_path)

    if args.mode == 5:
        show_layers(model_path, args.model_name)
