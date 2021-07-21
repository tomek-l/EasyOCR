import easyocr
reader = easyocr.Reader(['en']) # need to run only once to load model into memory
result = reader.readtext('/workdir/cstr-vedastr/IIIT5K/test/1002_2.png')
print(result)