# -*- coding: utf-8 -*-
'''
Created on Mar 28, 2018

@author: loitg
'''
import cv2
import numpy as np
import time
import random
import common
import multiprocessing
from multiprocessing import Process, Manager, Pool
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
import logging
from logging.handlers import TimedRotatingFileHandler
import datetime
from common import args
import json
from server_so import LocalServer as LocalServer_so
from server_chu import LocalServer as LocalServer_chu
# from pagepredictor import PagePredictor
from linepredictor import BatchLinePredictor
# from extract_fields.extract import CLExtractor
# from receipt import ReceiptSerialize, ExtractedData
import operator
app = Flask(__name__)
manager = None
server0 = None
server1 = None
logger = None

def createLogger(name):
    logFormatter = logging.Formatter("%(asctime)s [%(name)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger(name)
    
    fileHandler = TimedRotatingFileHandler(os.path.join(args.logsdir, 'log.' + name) , when='midnight', backupCount=10)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.DEBUG)
    return rootLogger

def runserver(server, states):
    logger = createLogger('server')
    server.run(states, logger) 


@app.route('/start')
def start():
    global manager,server0,server1, logger
    logger = createLogger('reader')
    manager = Manager()
    states = manager.dict()
    server0 = LocalServer_so(args.model_path, manager)
    print str(server0.queue_get)
    server1 = LocalServer_chu(args.model_path_chu, manager)
    print str(server1.queue_get)
    p = Process(target=runserver, args=(server0, states))
    p_chu = Process(target=runserver, args=(server1, states))
    p.daemon = True
    p_chu.daemon = True
    p.start()
    p_chu.start()
    return 'started: ' + str(p.pid) + ' and ' + str(p_chu.pid)

def combineString(pred_dict):
    rs = ''
    err = 0
    leng = 0
    for j in range(10):
        poss = set()
        for i in range(len(pred_dict)):
            if len(pred_dict[i]) > j:
                if pred_dict[i][j] != '-': poss.add(pred_dict[i][j])
        if len(poss) == 1:
            rs += list(poss)[0]
            leng += 1
        elif len(poss) > 1:
            rs += '(' + 'or'.join(poss) + ')'
            err += 1
            leng += 1
    print err , rs, leng
    return err , rs, leng

def readID(imgpath, logger):
    line_list = []
    line = cv2.imread(imgpath)
    line = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
    print 'DDDDDD' + str(server0)
    linereader = BatchLinePredictor(server0, logger)
    rows,cols = line.shape[:2]
    for i in range(args.batch_size):
        M = cv2.getRotationMatrix2D((cols/2,rows/2),random.randint(-1,1),1)
        aline = cv2.warpAffine(line,M,(cols,rows))
        padx = random.randint(int(0.5 * cols/12), int(1.0 * cols/12))
        pady = random.randint(int(0.5 * rows/12), int(1.0 * rows/12))
        if padx == 0: padx = 1
        if pady == 0: pady = 1
        aline = line[pady:-pady, padx:-padx]
        aline = aline * random.uniform(0.88,0.98)
        aline = aline.astype(np.uint8)
        newwidth = int(32.0/aline.shape[0] * aline.shape[1])
        if newwidth < 32 or newwidth > common.args.stdwidth: return 'Line too short or long'
        aline = cv2.resize(aline, (newwidth, 32))
        line_list.append(aline)
        cv2.imwrite('/tmp/' + str(i) + '.jpg', cv2.cvtColor(line, cv2.COLOR_RGB2BGR))
        
    batchname = datetime.datetime.now().isoformat()
    pred_dict = linereader.predict_batch(batchname, line_list, logger)
    multirs = []
    #with open('/tmp/output.csv', 'a') as of:
    #    of.write(imgpath.split('/')[-1] + ',' +  ','.join(pred_dict.values()) + '\n')
    #return ''
    multirs.append(combineString([pred_dict[0], pred_dict[1]]))
    multirs.append(combineString(['-' + pred_dict[0], pred_dict[1]]))
    multirs.append(combineString([pred_dict[0], '-' + pred_dict[1]]))
    multirs.sort()
    if multirs[0][0] < 3 and multirs[0][2]==9:
        return multirs[0][1]
    else:
        return 'Please try again'

def read(imgpath, logger, server):
    line_list = []
    line = cv2.imread(imgpath)
    line = cv2.cvtColor(line, cv2.COLOR_BGR2RGB)
    linereader = BatchLinePredictor(server, logger)

    newwidth = int(32.0/line.shape[0] * line.shape[1])
    if newwidth < 32 or newwidth > common.args.stdwidth: return 'Line too short or long'
    line = cv2.resize(line, (newwidth, 32))
    for i in range(args.batch_size):
        line_list.append(line)
        
    batchname = datetime.datetime.now().isoformat()
    pred_dict = linereader.predict_batch(batchname, line_list, logger)
    rs = pred_dict[0] if len(pred_dict[0].strip()) > 0 else '<blank line>'
    return rs  

class Line(object):
    def __init__(self,it,ot):
        self.inputtext = it
        self.outputtext = ot

def key2path(request, key):
    if (key not in request.files):
        return None
    f = request.files[key]
    if f.filename == '':
        return None
    if f:
        filename = secure_filename(f.filename)
        filename =  datetime.datetime.now().isoformat() \
                     + str(random.randint(1,99)) + '_' + key + '_' + filename
        imgpath = os.path.join('/home/loitg/cmnd_poc/ID/', filename)
        f.save(imgpath)
        return imgpath
    
    return None

def path2img2text(request, key, logger):
    imgpath = key2path(request, key)
    if imgpath is None:
        return ''
    
    #try:
        
    if key == 'lineID':
        return readID(imgpath, logger)
    elif key == 'lineNTNS':
        return read(imgpath, logger, server0)
    elif key in ['lineHoTen1', 'lineHoTen2', 'lineNguyenQuan1', 'lineNguyenQuan2']:
        return read(imgpath, logger, server1)
    
 #   except Exception:
  #      return 'Internal Error'
            
    return ''
    
@app.route('/ocr/cmnd9/lines', methods=['GET', 'POST'])
def upload_file():
    global logger
    if request.method == 'POST':
        rs = {}
        rs['textID'] = path2img2text(request, 'lineID', logger)
        rs['textHoTen1'] = path2img2text(request, 'lineHoTen1', logger)
        rs['textHoTen2'] = path2img2text(request, 'lineHoTen2', logger)
        rs['textNTNS'] = path2img2text(request, 'lineNTNS', logger)
        rs['textNguyenQuan1'] = path2img2text(request, 'lineNguyenQuan1', logger)
        rs['textNguyenQuan2'] = path2img2text(request, 'lineNguyenQuan2', logger)
        rs['textThuongTru1'] = path2img2text(request, 'lineThuongTru1', logger)
        rs['textThuongTru2'] = path2img2text(request, 'lineThuongTru2', logger)
        rs['fullImg'] = path2img2text(request, 'fullImg', logger)
        rs = json.dumps(rs).decode('utf-8')
        print rs
        return rs
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=lineID>
        <input type=file name=lineHoTen1>
        <input type=file name=lineHoTen2>
        <input type=file name=lineNTNS>
        <input type=file name=lineNguyenQuan1>
        <input type=file name=lineNguyenQuan2>
        <input type=file name=lineThuongTru1>
        <input type=file name=lineThuongTru2>
        <input type=file name=fullImg>
        <input type=submit value=Upload></p>
    </form>
    '''
            
if __name__ == '__main__':
    app.run(port=int("8080"), host="0.0.0.0")
    
    
    
