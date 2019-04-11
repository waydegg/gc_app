from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# model_file_url = 'https://www.dropbox.com/s/y4kl2gv1akv7y4i/stage-2.pth?raw=1'
# model_file_name = 'model'
model_file_url = 'https://drive.google.com/uc?export=download&id=1mqJytX0sLwOJhkGtDCdb8MVqW1qNpvjk'
model_file_name = 'center_text__final_gen'
classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

# # GAN Stuff
# PATH = Path('data')
# CLEAN = Path(PATH/'clean')
# MARKED = Path(PATH/'marked')
# src = ImageImageList.from_folder(MARKED).split_by_rand_pct(0.1, seed=42)

# def get_data(bs,size,src,clean_path):
#     tfms = get_transforms(do_flip=False, max_rotate=5.0, max_zoom=1.05, max_warp=0)
    
#     data = (src.label_from_func(lambda x: clean_path/x.name)
#            .transform(tfms, size=size, tfm_y=True)
#            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

#     data.c = 3
#     return data

# App Stuff
app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

async def setup_gan_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')



loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': str(learn.predict(img)[0])})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

