from config import API_key
import requests,os, io, time,random,mimetypes,sys
from datetime import datetime
from PIL import Image,ImageDraw,ImageFont

MODEL = "facebook/detr-resnet-50"
API = f"https://router.huggingface.co/models/{MODEL}"
ALLOWED, MAX_MB = {".png", ".jpg", ".jpeg",".bmp",".gif",".webp",".tiff"}, 8

EMOJI = {"person": "🧑","car": "🚗","truck": "🚚","bus": "🚌","bicycle": "🚲",
         "motorcycle" : "🏍️", "dog": "🐕","cat":"🐹","bird":"🐦","horse":"🐎","cow":"🐄","bear":"🐻"}

def font(sz=18):
    for f in ("DejaVuSans.ttf","arial.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except:
            pass
    return ImageFont.load_default()

def ask_image():
    print("\n Pick an Image (JPG, PNG, BMP, GIF, WEBP, TIFF < 8MB)")
    while True:
        p = input(" Image Path: ").strip('"').strip("'")
        if not p or not os.path.isfile(p):
            print("  -> Invalid file path, try again.")
            continue
        if os.path.splitext(p)[1].lower() not in ALLOWED:
            print("  -> Unsupported file type, try again.")
            continue
        if os.path.getsize(p) /(1024*1024) > MAX_MB:
            print("  -> File too large, try again.")
            continue
        try: Image.open(p).verify()
        except:
            print("  -> Corrupted or invalid image file, try again.")
            continue
        return p

def infer(path, img_bytes, tries=8):
    mime,_=mimetypes.guess_type(path)

    for _ in range(tries):
        if mime and mime.startswith("image/"):
            r = requests.post(API,
                                headers={"Authorization":f"Bearer {API_key}",
                                "Content-Type": mime},
                                data=img_bytes,timeout=60)
            
        else:
            r = requests.post(API,
                        headers={"Authorization":f"Bearer {API_key}"},
                                files = {"inputs":(os.path.basename(path),img_bytes,"application/octet-stream")},
                        timeout=60)
        if r.status_code == 200:
            d = r.json()
            if isinstance(d, dict) and "error" in d:
                raise RuntimeError(d["error"])
            if not isinstance(d, list):
                raise RuntimeError("Bad Api response")
            return d
        if r.status_code == 503:
            time.sleep(2); continue
        raise RuntimeError(f"API Error: {r.status_code}: {r.text[:300]}")
    raise RuntimeError("Model warm-up timeout")
def draw(img, dets,thr=0.5):
    d = ImageDraw.Draw(img); f = font(18); counts = {}
    
    for det in dets:
        s = float(det.get("score",0))
        if s < thr: continue

        lab = det.get("label","object")
        b = det.get("box",{})

        x1,y1,x2,y2 = (int(b.get(k,0)) for k in ("xmin","ymin","xmax","ymax"))
        if not (x2>0 and y2>0):
            x,y,w,h = int(b.get("x",0)), int(b.get("y",0)), int(b.get("w",0)), int(b.get("h",0))
            x1,y1,x2,y2 = x, y, x+w, y+h

        color = tuple(random.randint(80,255) for _ in range(3))
        d.rectangle([(x1,y1),(x2,y2)],outline=color,width=4)

        txt = f"{EMOJI.get(lab.lower(), '✨')} {lab} {s*100:0.f}%"
        tw,th = d.textlength(txt, font=f), f.size+6
        d.rectangle([(x1,max(0, y1-th)),(x1+tw + 8, y1)], fill=color)
        d.text((x1+4, y1-th+3), txt, fill=(0,0,0))
        counts[lab] = counts.get(lab,0) + 1       
    return counts

def main():
    path = ask_image()

    with open(path,"rb") as fh: by = fh.read()

    try: dets = infer(path, by)
    except Exception as e:
        print(f" Error: {e}")
        return
    
    img = Image.open(io.BytesIO(by)).convert("RGB")

    counts = draw(img, dets, thr=0.5)

    out = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    img.save(out); print(f"\n Saved annotated image as: {out}")

    if counts:
        print("\n Detected Objects:")
        for k,v in sorted(counts.items(),key=lambda kv:(-kv[1],kv[0])):
            print(f"●{EMOJI.get(k.lower(),'✨')} {k}: {v}")
    
    else:
        print("\n No objects detected --- try a clearer or busier scene")

    print("\n ⚠ Disclamer: This is an experimental model, results may vary.")
if __name__ == "__main__":
    main()