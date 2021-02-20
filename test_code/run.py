from cartoonize import WhiteBox
import argparse
import os

def validate_path(fname, is_file=True, exts=None):
    if not os.path.exists(fname):
        raise Exception(f"{fname} does not exist")
    if is_file:
        if not os.path.isfile(fname):
            raise Exception(f"{fname} is not a valid file.")
    else:
        if not os.path.isdir(fname):
            raise Exception(f"{fname} is not a valid directory.")
    if is_file and exts:
        root, ext = os.path.splitext(fname)
        # remove trailing '.' in extention.
        clean_ext = ext[1:]
        if clean_ext not in exts:
            raise Exception(f"{fname} is not a valid type. Valid file types are: {exts}")

def file_helper(args):
    model_path = os.path.abspath(args.model_path)
    validate_path(model_path, is_file=False)
    #
    out_path = os.path.abspath(args.output)
    if not os.path.exists(out_path) and not args.dont_save:
         os.mkdir(out_path)
    validate_path(out_path, is_file=False)
    #
    if args.image_path:
        fpath = os.path.abspath(args.image_path)
        validate_path(fpath, exts=['jpeg', 'jpg', 'png'])
        fname = os.path.basename(fpath).split('.')[0]
        save_path = None if args.dont_save else os.path.join(out_path, f'{fname}.jpg')
    elif args.video_path:
        fpath = os.path.abspath(args.video_path)
        validate_path(fpath, exts=['mp4', 'mov'])
        fname = os.path.basename(fpath).split('.')[0]
        save_path = None if args.dont_save else os.path.join(out_path, f'{fname}.mov')
    else:
        raise Exception("Missing Video/Image source. Must provide either video or image path.")
    return model_path, save_path, fpath


def run(args):
    try:
        model, save_file, load_file = file_helper(args)
        # print(model, save_file, load_file)
        wb = WhiteBox(model_path=model, save_path=save_file)
        if args.image_path:
            wb.cartoonize_image(load_file, args.quite_mode)
        elif args.video_path:
            wb.cartoonize_video(load_file, args.quite_mode)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    # if not os.path.exists(save_folder):
    #     os.mkdir(save_folder)
    parser = argparse.ArgumentParser(description='Cartoonize video and images.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m',
        '--model-path',
        help="Path to the model location.",
        type=str,
        default="./saved_models"
        )
    parser.add_argument(
        '-o',
        '--output',
        help="Location to save the output file.",
        type=str,
        default="./cartoonized_output"
        )
    parser.add_argument(
        '-D',
        '--dont-save',
        help="Runs program without saving the output file.",
        action='store_true',
        default=False
        )
    parser.add_argument(
        '-q',
        '--quite-mode',
        help="Runs program without diplaying output.",
        action='store_true',
        default=False
        )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '-i',
        '--image-path',
        help="Path to the image location to be processed.",
        type=str,
        default=None
        )
    group.add_argument(
        '-v',
        '--video-path',
        help="Path to the video location to be processed.",
        type=str,
        default=None
        )
    # Parse user input to match flags.
    args = parser.parse_args()
    # Run functions.
    print(args)
    run(args)

    