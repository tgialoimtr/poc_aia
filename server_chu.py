'''
Created on Feb 19, 2018

@author: loitg
'''
# This is a placeholder for a Google-internal import.
import cv2
from time import sleep, time
from Queue import Empty, Full

import numpy as np
import tensorflow as tf

from weinman import model_chu as model
from weinman import mjsynth, validate
from common import args

class Bucket(object):
    def __init__(self, maxtime, batchsize, widthrange):
        self.maxtime = maxtime
        self.batchsize = batchsize
        self.widthrange = widthrange
        self.imgs = []
        self.widths = []
        self.infos = []
        self.oldesttime = None

    
    def addImgToBucket(self, clientid, imgid, imgtime, img):
        if img.shape[1] > self.widthrange[0] and img.shape[1] <= self.widthrange[1]:
            newimg = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=self.widthrange[1]-img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
#             if len(newimg.shape) < 3:
#                 newimg = newimg[:,:,np.newaxis]
#             else:
#                 newimg = newimg[:,:,1]
            self.imgs.append(newimg)
            self.widths.append(img.shape[1])
            self.infos.append((clientid, imgid))
            if self.oldesttime is None or imgtime < self.oldesttime:
                self.oldesttime = imgtime
            return True
        else:
            return False
    
    
    def getBatch(self):
        if len(self.imgs) == 0: return None
        if len(self.imgs) > self.batchsize or (time() - self.oldesttime) > self.maxtime:
            batch = np.array(self.imgs[:self.batchsize])
            widths = np.array(self.widths[:self.batchsize])
            infos = self.infos[:self.batchsize]
            self.imgs = self.imgs[self.batchsize:]
            self.widths = self.widths[self.batchsize:]
            self.infos = self.infos[self.batchsize:]
            self.oldesttime = time() #fix this maybe
            return infos, batch, widths
        else:
            return None

class LocalServer(object):
    def __init__(self, modeldir, manager):
        self.manager = manager
        self.modeldir = modeldir
        self.queue_get = self.manager.Queue()
        self.queue_push = self.manager.Queue()
        
#         self.client_inputs = {}
#         self.client_outputs = {}
#         self.buckets = []
#         for w in range(32, 1000,32):
#             self.buckets.append(Bucket(args.bucket_max_time, args.bucket_size,(w,w+32)))
#         self.graph = None
#         self.maxclientid = 0
#         self.modeldir = modeldir
#         self.manager = manager
#     
#     def register(self):
#         clientid = str(self.maxclientid)
#         self.maxclientid += 1
#         self.client_inputs[clientid] = self.manager.Queue()
#         self.client_outputs[clientid] = self.manager.Queue()
#         return clientid, self.client_inputs[clientid], self.client_outputs[clientid]
        
    def run(self, states, logger):
        with tf.Graph().as_default():
            image,width = validate._get_input(None) # Placeholder tensors
 
            proc_image = validate._preprocess_image(image)

            with tf.device(args.device):
                features,sequence_length = model.convnet_layers( proc_image, width, 
                                                                 validate.mode)
                logits = model.rnn_layers( features, sequence_length,
                                           mjsynth.num_classes_chu() )
                predictions = validate._get_output( logits,sequence_length)
            logger.info('AAAAAAA')
            session_config = validate._get_session_config()
            restore_model = validate._get_init_trained()
            logger.info('BBBBBBB')
            init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 
            with tf.Session(config=session_config) as sess:
                logger.info('CCCCCCC')
                sess.run(init_op)
                restore_model(sess, validate._get_checkpoint(self.modeldir)) # Get latest checkpoint
                logger.info('server started, waiting image ...')
#                 print(str(time()) + 'server started, waiting image ...') 
#                 states['server_started'] = True
                try:
                    while True:
                        imglist = []
                        batchid = None
                        while True:
                            try:
                                batchid, imglist = self.queue_get.get(block=False)
                                if len(imglist) > 0: break
                            except Empty:
    #                             print(str(time()) + ': queue put ' + clientid + ' empty')
                                sleep(0.3)
                        forimg = []
                        forwidth = []
                        
                        for img in imglist:
                            if img.shape[1] > args.stdwidth: continue
                            newimg = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=args.stdwidth-img.shape[1], borderType=cv2.BORDER_CONSTANT, value=0)
#                             if len(newimg.shape) < 3:
#                                 newimg = newimg[:,:,np.newaxis]
#                             else:
#                                 newimg = newimg[:,:,1]
                            forimg.append(newimg)
                            forwidth.append(img.shape[1])
                        
                        batch = np.array(forimg)
                        forwidth = np.array(forwidth)
                        p = sess.run(predictions,{ image: batch, width: forwidth} )
                        fortxt = []
                        for j in range(p[0].shape[0]):
                            txt = p[0][j,:]
                            txt = [i for i in txt if i >= 0]
                            txt = validate._get_string_chu(txt) 
                            fortxt.append(txt)
                        try:
                            self.queue_push.put((batchid, fortxt), block=False)
                        except Full:
                            logger.warning('queue get full')
                except Exception:
                    logger.exception('SERVER ERROR')   
    
if __name__ == '__main__':
    pass
