'''
Common arguments and parameters
'''
class DFO(object):
    pass

args = DFO()
args.model_path = '/home/loitg/workspace/poc_aia_resources/model_id-so/'
args.model_path_chu = '/home/loitg/workspace/poc_aia_resources/model_chu3/'
args.imgsdir = '/home/loitg/ctoaia/case1_line/hard_imgs/' # test image for validate.py
args.numprocess = 1 #CPU:2 #GPU:8
args.qget_wait_count = 400000
args.qget_wait_interval = 0.3
args.stdwidth=32*20
args.bucket_size = 1 #CPU:2 #GPU:16 
args.batch_size = 2
args.bucket_max_time = 10
args.device = '/device:CPU:0'
args.logsdir = '/home/loitg/location_nn/logs/' # store logs files
