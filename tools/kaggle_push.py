# Запуск ядра Kaggle через REST: KT=<токен в окружении, не в файлах> python tools/kaggle_push.py <папка с kernel-metadata.json>
import json,sys,os,urllib.request
meta=json.load(open(sys.argv[1]+'/kernel-metadata.json')); code=open(sys.argv[1]+'/'+meta['code_file'],encoding='utf-8').read()
body=dict(slug=meta['id'],newTitle=meta['title'],text=code,language='python',kernelType=meta['kernel_type'],isPrivate=True,enableGpu=bool(meta.get('enable_gpu')),machineShape=meta.get('machine_shape',''),enableTpu=False,enableInternet=True,
          datasetDataSources=meta.get('dataset_sources',[]),kernelDataSources=meta.get('kernel_sources',[]),competitionDataSources=[],modelDataSources=[],categoryIds=[])
r=urllib.request.Request('https://www.kaggle.com/api/v1/kernels/push',json.dumps(body).encode(),{'Authorization':'Bearer '+os.environ['KT'],'Content-Type':'application/json'})
try: print(urllib.request.urlopen(r).read().decode()[:600])
except urllib.error.HTTPError as e: print(e.code,e.read().decode()[:600])
