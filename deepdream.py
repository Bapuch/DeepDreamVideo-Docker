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
import caffe
import shutil
import os
import sys
import random
import tempfile
import time

from subprocess import Popen, PIPE
import errno


# -- Argument Parsing
def get_parser():
    parser = argparse.ArgumentParser(description="DeepDream Video with Caffe")

    parser.add_argument(
        '-i','--input',
        type=str,
        help='Input directory where extracted frames are stored',
        required='--extract' not in sys.argv and '-e' not in sys.argv,
        default='/data/input_frames',
        dest='output_dir')

    parser.add_argument(
        '-e','--extract',
        type=str,
        help='Path to video to process',
        required='--input' not in sys.argv and '-i' not in sys.argv,
        dest='extract',)

    parser.add_argument(
        '-o','--output',
        type=str,
        help='Output directory where processed frames are to be stored',
        default='/data/output_frames',
        dest='input_dir')



    parser.add_argument(
        '-it','--image_type',
        help='Specify whether frames will be jpg or png ',
        default='jpg',
        type=str,
        required=False,
        dest='image_type')

    parser.add_argument(
        '-p', '--model_path',
        type=str,
        dest='model_path',
        required=False,
        default='../caffe/models/bvlc_googlenet/',
        help='Model directory to use')
    parser.add_argument(
        '-m', '--model_name',
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
        dest='octavescale',)
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
        dest='stepsize',)
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
        # type=str,
        required=False,
        help='Array of Layers to loop through. Default: ["customloop"] \
        - or choose ie ["inception_4c/output"] for that single layer',
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
        '-gi', '--guide_image',
        required=False,
        type=str,
        help="path to guide image",
        default=None,
        dest='guide_image',)
    parser.add_argument(
        '-sf', '--start_frame',
        type=int,
        required=False,
        help="starting frame number",
        default=1,
        dest='start_frame',)
    parser.add_argument(
        '-ef', '--end_frame',
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


def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
              end='inception_4c/output', clip=True, image_output=None,
              save_image=None, **step_params):
 
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
            if save_image is not None and image_output is not None: #and i==iter_n-1
                image_file = f"{image_output}/o{octave}_i{i}_{end.replace('/', '-')}-{difference}s{save_image}"

                # The keys have directory / in them, may need to mkdir
                image_dir = os.path.dirname(image_file)
                if not os.path.exists(image_dir):
                    # not entirely safe way to do it, but ok for start_time
                    os.makedirs(image_dir)

                PIL.Image.fromarray(np.uint8(vis)).save(image_file)

            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            # showarray(vis)
            print("Octave:", octave, " - Iter:", i, " - Layer:", end, " - Shape:", vis.shape)
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base

    # returning the resulting image
    return deprocess(net, src.data[0])

def objective_guide(dst):
    x = dst.data[0].copy()
    y = dst.data[0].copy()
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    
    
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
# def main():
def main(input_dir, output, image_type, model_path, model_name, octaves, octave_scale,
         iterations, jitter, stepsize, blend, layers, guide_image, start_frame, end_frame, verbose,
        base_name):






#     tmpdir = tempfile.mkdtemp()
#     models_dir = get_envar('DEEPDREAM_MODELS', args.models_dir)
#     frames = int(os.environ.get('DEEPDREAM_FRAMES', args.frames))
#     s = float(os.environ.get('DEEPDREAM_SCALE_COEFF', args.s)) # scale coefficient
    scale_coef = 0.5
#     image_dir = os.environ.get('DEEPDREAM_IMAGES', args.image_dir)
#     image_output = args.image_output or os.environ.get('DEEPDREAM_OUTPUT', tmpdir)
#     image_input =  os.environ.get('DEEPDREAM_INPUT', args.input) or '/deepdream/deepdream/sky1024px.jpg'


    make_sure_path_exists(input_dir)
    make_sure_path_exists(output)
    
     # let max nr of frames
    nrframes =len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
    if nrframes == 0:
        print("no frames to process found")
        sys.exit(0)
    
#     # -- Input Checking
                        
#     if not os.path.exists(image_output):
#         os.makedirs(image_output)

#     if not os.path.exists(image_input):

#         # Second try - user mounted to data, but image is in $PWD
#         image_input = "/data/%s" % image_input

#         if not os.path.exists(image_input):
#             print('Cannot find %s.' % image_input)
#             sys.exit(1)

#     lookup = find_model(models_dir, 'bvlc_googlenet')

#     # -- Loading DNN Model

#     # Patching model to be able to compute gradients.
#     # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
#     model = caffe.io.caffe_pb2.NetParameter()
#     text_format.Merge(open(lookup['net_fn']).read(), model)
#     model.force_backward = True

#     tmp_proto = '%s/tmp.prototxt' % tmpdir
#     open(tmp_proto, 'w').write(str(model))

#     net = caffe.Classifier(tmp_proto, lookup['param_fn'],
#                            mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
#                            channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    
    #Load DNN
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + model_name #

    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    
    # --- Dream!
    start_time = time.time()
    totaltime = 0

#     input_name = os.path.basename(image_input)
#     img = np.float32(PIL.Image.open(image_input))
    

#     dreamy = deepdream(net, img)

#     PIL.Image.fromarray(np.uint8(dreamy)).save("%s/dreamy-%s" % (image_output, input_name))
    
#     later = time.time()
#     difference = int(later - start_time)
#     totaltime += difference
#     print('BASE FRAME ')
#     print('>> Frame Time: ' + str(difference) + 's  ---  Total Time: ' + str(difference) + 's')
#     print('***************************************')
    
    if start_frame is None:
        frame_i = 1
    else:
        frame_i = int(start_frame)
    if not end_frame is None:
        nrframes = int(end_frame)+1
    else:
        nrframes = nrframes+1
        
    frame = np.float32(PIL.Image.open(input_dir + '/%08d.%s' % (frame_i, image_type) ))
    
#     # --- With Guide?
#     if args.guide is not None:
#         frame_start_time = time.time()

#         guide = np.float32(PIL.Image.open(args.guide))
#         end = 'inception_3b/output'
#         h, w = guide.shape[:2]
#         src, dst = net.blobs['data'], net.blobs[end]
#         src.reshape(1,3,h,w)
#         src.data[0] = preprocess(net, guide)
#         net.forward(end=end)
#         guide_features = dst.data[0].copy()
#         guided = deepdream(net, img, end=end, objective=objective_guide)
#         print('GUIDE FRAME : saving..')
#         PIL.Image.fromarray(np.uint8(guided)).save("%s/guided-%s" % (image_output, input_name))
#         img = guided

#         later = time.time()
#         difference = int(later - start_time)
#         totaltime += difference
        
#         print('>> GUIDE Frame Time: ' + str(int(later - frame_start_time)) + 's  ---  Total Time: ' + str(difference) + 's')
#         print('***************************************')

    # TODO: net.blobs.keys() we can change layer selection to alter the result! 

#     frame = img
#     frame_i = 0

    h, w = frame.shape[:2]
#     for i in range(frames):
    print('nb frames: ', nrframes-1)
    for i in range(frame_i, nrframes):
        
        #Choosing Layer
        if layers == 'customloop': #loop over layers as set in layersloop array
            endparam = layersloop[frame_i % len(layersloop)]
        else: #loop through layers one at a time until this specific layer
            endparam = layers[frame_i % len(layers)]
            
            
        frame_start_time = time.time()

        frame = deepdream(net, frame, end=endparam, image_output=output + f"/{base_name}", save_image=f".{image_type}")
#         PIL.Image.fromarray(np.uint8(frame)).save("%s/frame-%04d-%s" % (image_output, frame_i, input_name))
        later = time.time()
        difference = int(later - start_time)    
        PIL.Image.fromarray(np.uint8(frame)).save(output + f"/{base_name}-{difference}s_{i:08d}.{image_type}")
        
        

        
        totaltime += difference
        avgtime = (totaltime / (i))
        print('FRAME ' + str(i) + ' of ' + str(nrframes-1))
        print('>> Frame Time: ' + str(int(later - frame_start_time)) + 's  ---  Total Time: ' +
              str(difference) + 's')
        timeleft = avgtime * ((nrframes-1) - frame_i)        
        m, s = divmod(timeleft, 60)
        h, m = divmod(m, 60)
        print('>> Estimated Total Time Remaining: ' + str(timeleft) +
              's (' + "%d:%02d:%02d" % (h, m, s) + ')')
        frame = nd.affine_transform(frame, [1-scale_coef,1-scale_coef,1], [h*scale_coef/2,w*scale_coef/2,0],
                                    order=1)
        frame_i += 1
        print('***************************************')
        
    print('DeepDreams are made of cheese, who am I to diss a brie?')
#     print('output> %s' % image_output)

    # Remove temporary parameters, we could keep this if someone wanted
#     shutil.rmtree(tmpdir)

def extract_video(video, ext, frame_dir):
    # output_dir = _output_video_dir(video)
    # mkdir(output_dir)
    # output = Popen(
    #     "ffmpeg -loglevel quiet -i {} -f image2 {}/img_%4d.jpg".format(
    #         video, output_dir), shell=True, stdout=PIPE).stdout.read()
    make_sure_path_exists(frame_dir)
    print(Popen(f'ffmpeg -i {video} -f image2 {frame_dir}/%08d.{ext}', shell=True,
                           stdout=PIPE).stdout.read())

    print(f'Frames created to {frame_dir}')

def create_video(frames_directory, original_video, ext, output_video,frame_rate=24):
    # make_sure_path_exists(frame_dir)

    # output = Popen((
    #     f"ffmpeg -loglevel quiet -r {frame_rate} -f image2 -pattern_type glob -i {output_dir}/* {video}.mp4"),
    #     shell=True, stdout=PIPE).stdout.read()
    
    output = Popen((
        f"./frames2movie.sh ffmpeg {frames_directory} {original_video} {ext} {output_video}"),
        shell=True, stdout=PIPE).stdout.read()
        # "./3_frames2movie.sh [ffmpeg|avconv|mplayer] [frames_directory] [original_video_with_sound] [png|jpg]"
    print("OUTPUT =", output)


if __name__ == '__main__':
    # main(input_dir, output_dir, image_type, model_path, model_name, preview, octaves, octave_scale,
    #      iterations, jitter, zoom, stepsize, blend, layers, guide_image, start_frame, end_frame, verbose,
    #     base_name)

    parser = get_parser()

    def help(return_code=0):
        parser.print_help()
        sys.exit(return_code)
    
    # If the user didn't provide any arguments, show the full help
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    if not os.path.exists(args.model_path):
        print("Model directory not found")
        print("Please set the model_path to a correct caffe model directory")
        sys.exit(0)

    model = os.path.join(args.model_path, args.model_name)
    print(model)
    if not os.path.exists(model):
        print("Model not found")
        print("Please set the model_name to a correct caffe model")
        print("or download one with for instance: ./caffe_dir/scripts/download_model_binary.py caffe_dir/models/bvlc_googlenet")
        sys.exit(0)

    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.extract:
        
        video_name, _ = os.path.splitext(os.path.basename(args.extract))
        input_dir = os.path.join(input_dir, video_name)
        output_dir = os.path.join(output_dir, video_name)
        # extract_video(args.extract, args.image_type, input_dir)

        t = str(time.time()).replace('.', '_')
        videos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/dream_video")
        make_sure_path_exists(videos_dir)
        output_video = os.path.join(videos_dir, f"{video_name}-DeepDream__{t}")

        create_video(input_dir, args.extract, args.image_type, output_video)

    
    # main(input_dir=input_dir, 
    #     output_dir=output_dir, 
    #     image_type=args.image_type, 
    #     model_path=args.model_path, 
    #     model_name=args.model_name, 
    #     octaves=args.octaves, 
    #     octavescale=args.octavescale, 
    #     iterations=args.iterations, 
    #     jitter=args.jitter,
    #     stepsize=args.stepsize, 
    #     blend=args.blend, 
    #     layers=args.layers, 
    #     guide_image=args.guide_image, 
    #     start_frame=args.start_frame, 
    #     end_frame=args.end_frame, 
    #     verbose=args.verbose)

