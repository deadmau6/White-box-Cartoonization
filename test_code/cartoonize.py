import os
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import network
import guided_filter
from tqdm import tqdm

class WhiteBox:

    def __init__(self, model_path='./saved_models', save_path=None, auto_load=True):
        self.model_path = os.path.abspath(model_path)
        self.save_path = os.path.abspath(save_path)
        if auto_load:
            self.load_model()
            self.loaded = True

    @staticmethod
    def resize_crop(image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                           interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        return image

    def load_model(self):
        try:
            tf.disable_eager_execution()
        except:
            None
        tf.reset_default_graph()
        #
        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)
        #
        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)
        #
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        #
        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))

    def _cartoonize(self, img):
        image = self.resize_crop(img)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output

    def cartoonize_image(self, image_path, quite_mode=False):
        # load_path = os.path.join(image_path, name)
        image = cv2.imread(image_path)
        cartoon = self._cartoonize(image)
        if self.save_path:
            cv2.imwrite(self.save_path, cartoon)
        if not quite_mode:
            cv2.imshow('Cartoonized', cartoon)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

    def cartoonize_video(self, video_source, quite_mode=False):
        # load_path = os.path.abspath(video_source)
        video = cv2.VideoCapture(video_source)
        save_video = False
        size = (512,512)
        output = None
        #
        if self.save_path:
            out_path = os.path.abspath(self.save_path)
            size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = video.get(cv2.CAP_PROP_FPS)
            print(fps, size)
            output = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
            save_video = True
        #
        if not quite_mode:
            cv2.namedWindow("cam-input")
            cv2.namedWindow("cam-output")
        while True:
            success, frame = video.read()
            if not success:
                print('End of video stream.')
                break
            if not quite_mode:
                cv2.imshow("cam-input", frame)
            #
            cartoon_frame = self._cartoonize(frame)
            #
            if save_video:
                output.write(cartoon_frame)
            if not quite_mode:
                cv2.imshow("cam-output", cartoon_frame)
            #
            k = cv2.waitKey(1)
            if k == 27 or k == ord('q'):
                break
        video.release()
        if save_video:
            output.release()
        cv2.destroyAllWindows()