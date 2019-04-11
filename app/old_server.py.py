from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# GAN Stuff
PATH = Path('data')
CLEAN = Path(PATH/'clean')
MARKED = Path(PATH/'marked')
src = ImageImageList.from_folder(MARKED).split_by_rand_pct(0.1, seed=42)

def get_data(bs,size,src,clean_path):
    tfms = get_transforms(do_flip=False, max_rotate=5.0, max_zoom=1.05, max_warp=0)
    
    data = (src.label_from_func(lambda x: clean_path/x.name)
           .transform(tfms, size=size, tfm_y=True)
           .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data

export_file_url = 'https://drive.google.com/uc?export=download&id=1tghKa7V_DEC9-X1eBJEbBN_zJZzKaatu'
export_file_name = 'export.pkl'

# classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

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
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

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
    
    # Getting a single prediction
    img = open_image(BytesIO(img_bytes))
    h,w = img.size
    if w % 2 != 0:
        w += 1
    if h % 2 != 0:
        h += 1
    learn.data = get_data(1, (h,w), src, CLEAN)
    prediction = learn.predict(img)[0]
    prediction.save(path/'data'/'demarked_img.png')

    return JSONResponse({'result'})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
