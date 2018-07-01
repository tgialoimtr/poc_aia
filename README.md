# poc_aia
This is a REST interface for read individual lines of CMND 9 so of VietNam.


# install dependencies:
pip install ...
pip install flask
pip install tensorflow==1.4.0 
(recommend 1.4.0 or 1.5.0, tensorflow or tensorflow_gpu both fine)

# run:
python webapp.py
from browser, access http://localhost:8080/start to trigger start controller to start tensorflow server. Sometimes it hangs, refresh page.
from browser, access http://localhost:8080/ocr/cmnd9/lines to view GUI interface to upload individual lines, press "Upload" and wait for json text result
