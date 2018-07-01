# poc_aia
This is a REST interface for read individual lines of CMND 9 so of VietNam.  


# install dependencies:
pip install ...  
pip install flask  
pip install tensorflow==1.4.0   
(recommend 1.4.0 or 1.5.0, tensorflow or tensorflow_gpu both fine)  

# run:
Edit common.py:
1. args.model_path = '/home/loitg/workspace/poc_aia_resources/model_id-so/' ==> pre-trained model for so, ngay thang nam sinh
2. args.model_path_chu = '/home/loitg/workspace/poc_aia_resources/model_chu3/' ==> pre-trained model for ten, que quan
3. args.logsfile ==> store log files   


python webapp.py  
from browser, access http://localhost:8080/start to trigger start controller to start tensorflow server. Sometimes it hangs, refresh page.  
from browser, access http://localhost:8080/ocr/cmnd9/lines to view GUI interface to upload individual lines, press "Upload" and wait for json text result  
