from argparse import Namespace

from data.base_dataset import get_params, get_transform

from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from PIL import Image

from fastapi import FastAPI, Body
from starlette.requests import Request
import uvicorn
import base64


# print("hohoo",opt)
opt = Namespace()
opt.batchSize=1
opt.data_type=32
opt.dataroot='dataset/'
opt.display_winsize=512
opt.fineSize=512
opt.gen_checkpoint='checkpoints/PFAFN/gen_model_final.pth'
opt.gpu_ids=[]
opt.input_nc=3
opt.isTrain=False
opt.loadSize=512
opt.max_dataset_size="inf"
opt.nThreads=1
opt.name='demo'
opt.no_flip=False
opt.norm='instance'
opt.output_nc=3
opt.phase='test'
opt.resize_or_crop='None'
opt.serial_batches=False
opt.tf_log=False
opt.use_dropout=False
opt.verbose=False
opt.warp_checkpoint='checkpoints/PFAFN/warp_model_final.pth'
device = torch.device("cpu")

warp_model = AFWM(opt, 3)
# print(warp_model)
warp_model.eval()
# warp_model.to(device)
warp_model.to(device)
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
# print(gen_model)
gen_model.eval()
gen_model.to(device)
load_checkpoint(gen_model, opt.gen_checkpoint)

def tryon(per_img,clo_img):
    # t1 = time.time()
    im_bytes_per = base64.b64decode(per_img)
    im_arr_per = np.frombuffer(im_bytes_per, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img_per = cv2.imdecode(im_arr_per, flags=cv2.IMREAD_COLOR)

    im_bytes_clo = base64.b64decode(clo_img)
    im_arr_clo = np.frombuffer(im_bytes_clo, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img_clo = cv2.imdecode(im_arr_clo, flags=cv2.IMREAD_COLOR)
    # print(img_per)
    # clothes_ben = Image.open("dataset/test_clothes/003434_1.jpg").convert('RGB')
    clothes_ben = cv2.cvtColor(cv2.resize(img_clo,(192,256)), cv2.COLOR_BGR2RGB)
    clothes_ben = Image.fromarray(clothes_ben)
    params = get_params(opt, clothes_ben.size)
    transform = get_transform(opt, params)
    transform_E = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    clothes_ben_tensor = transform(clothes_ben)

    # person_ben = Image.open("dataset/test_img/4.jpg").convert('RGB').resize([192,256])
    person_ben = cv2.cvtColor(cv2.resize(img_per,(192,256)), cv2.COLOR_BGR2RGB)
    person_ben = Image.fromarray(person_ben)
    person_ben_tensor = transform(person_ben)

    # edge_ben = Image.open("dataset/test_edge/003434_1.jpg").convert('L')
    mask = np.zeros(img_clo.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    h,w = img_clo.shape[:2]
    rect = (1,1,w,h)
    # print(rect)
    cv2.grabCut(img_clo,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    edge_ben = mask2[:,:,np.newaxis]*255
    edge_ben_tensor = transform_E(edge_ben)

    real_image = person_ben_tensor.expand(1,-1,-1,-1)
    # print("ÔIUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU",real_image)
    clothes = clothes_ben_tensor.expand(1,-1,-1,-1)
    ##edge is extracted from the clothes image with the built-in function in python
    edge = edge_ben_tensor.expand(1,-1,-1,-1)

    edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
    clothes = clothes * edge        

    flow_out = warp_model(real_image.to(device), clothes.to(device))
    warped_cloth, last_flow, = flow_out
    warped_edge = F.grid_sample(edge.to(device), last_flow.permute(0, 2, 3, 1),
                    mode='bilinear', padding_mode='zeros')

    gen_inputs = torch.cat([real_image.to(device), warped_cloth, warped_edge], 1)
    gen_outputs = gen_model(gen_inputs)
    p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
    p_rendered = torch.tanh(p_rendered)
    m_composite = torch.sigmoid(m_composite)
    m_composite = m_composite * warped_edge
    p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    cv_img=(p_tryon.squeeze().permute(1,2,0).detach().cpu().numpy()+1)/2
    rgb=(cv_img*255).astype(np.uint8)
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    # cv2.imwrite('ress.jpg',bgr)
    # print("haaaaaaaaaaaaaaaaaaaaaaaa",time.time()-t1)
    _, im_arr = cv2.imencode('.jpg', bgr)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

app = FastAPI()

@app.post("/api")
def try_on(request:Request,per_img: str = Body(...),clo_img: str = Body(...)):
    if request.method == "POST":  
        results = {}
        img = tryon(per_img,clo_img)
        results['img']= img  
        return {"status":"200","messageCode":"api.success","message":"API OK","results": results}


if __name__ == "__main__":
    uvicorn.run(app,debug=True,host="0.0.0.0",port=8000)


