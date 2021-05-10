#!/usr/bin/env python
import os
import sys
import time
import yaml
import hashlib
import argparse

from six.moves import urllib

# required_keys = ['caffemodel', 'caffemodel_url', 'sha1']
required_keys = ['caffemodel', 'caffemodel_url']


from caffe.io import caffe_pb2
from caffe import Classifier
from google.protobuf import text_format

def get_layers(model_path, model_name):
    net_fn   = os.path.join(model_path, 'deploy.prototxt')
    param_fn = os.path.join(model_path, model_name)

    model = caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = Classifier('tmp.prototxt', param_fn) # the reference model has channels in BGR order instead of RGB
    print("LAYERS for model " + model_name)
    print()
    for k in net.blobs:
        print(k)

def reporthook(count, block_size, total_size):
    """
    From http://blog.moleculea.com/2012/10/04/urlretrieve-progres-indicator/
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = (time.time() - start_time) or 0.01
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def parse_readme_frontmatter(dirname):
    readme_filename = os.path.join(dirname, 'readme.md')
    with open(readme_filename) as f:
        lines = [line.strip() for line in f.readlines()]

    
    top = lines.index('---')

    bottom = lines.index('---', top + 1)

    frontmatter = yaml.load('\n'.join(lines[top + 1:bottom]))

    missing_keys = [key for key in required_keys if key not in frontmatter ]
    if missing_keys:
        print("ERROR, key missing in readme:")
        print(missing_keys)

    assert all(key in frontmatter for key in required_keys)
    return dirname, frontmatter


def valid_dirname(dirname):
    try:
        return parse_readme_frontmatter(dirname)
    except Exception as e:
        print('ERROR: {}'.format(e))
        raise argparse.ArgumentTypeError(
            'Must be valid Caffe model directory with a correct readme.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download trained model binary.')
    parser.add_argument('dirname', type=valid_dirname)
    args = parser.parse_args()

    # A tiny hack: the dirname validator also returns readme YAML frontmatter.
    dirname = args.dirname[0]
    frontmatter = args.dirname[1]
    model_filename = os.path.join(dirname, frontmatter['caffemodel'])

    # Closure-d function for checking SHA1.
    def model_checks_out(filename=model_filename, sha1=frontmatter.get('sha1')):
        with open(filename, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest() == sha1

    # Check if model exists.
    if os.path.exists(model_filename):
        if frontmatter.get('sha1'):
            if model_checks_out():
                print("Model already exists.")
                sys.exit(0)

        else:
            print("Model already exists.")
            sys.exit(0)

    # Download and verify model.
    try:
        (filename, headers) = urllib.request.urlretrieve(
        frontmatter['caffemodel_url'], model_filename, reporthook)
    except Exception as e:
        raise RuntimeError("Failed to download '{}'.\n'{}'".format(frontmatter['caffemodel_url'], e))
    
    if frontmatter.get('sha1'):
        if not model_checks_out():
            print('ERROR: model did not download correctly! Run this again.')
            sys.exit(1)

    # Show layers:
    get_layers(dirname, frontmatter['caffemodel'])


