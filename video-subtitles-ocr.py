# %%
# 0. 下载语言

import os
from urllib.request import urlretrieve


def download_tessdata(url, savepath='./'):
    # 显示下载进度
    def reporthook(a, b, c):
        print("\rdownloading: %5.1f%%" % (a * b * 100.0 / c), end="")

    filename = os.path.basename(url)
    if not os.path.isfile(os.path.join(savepath, filename)):
        print('Downloading data from %s' % url)
        urlretrieve(url, os.path.join(savepath, filename), reporthook=reporthook)
        print('\nDownload finished!')
    else:
        print('File already exsits!')

    filesize = os.path.getsize(os.path.join(savepath, filename))  # 获取文件大小
    print('File size = %.2f Mb' % (filesize / 1024 / 1024))  # Bytes转换为Mb


tessdata_dir = './tessdata/'
tessdata_url = 'https://ghproxy.com/https://raw.githubusercontent.com/tesseract-ocr/tessdata/master/{}.traineddata'

# 语言： 中+英
lang = 'chi_sim+eng'
for lang_name in lang.split('+'):
    download_tessdata(tessdata_url.format(lang_name), tessdata_dir)

# %%
# 1. 读取视频
import cv2

video_path = 'd7.mp4'

v = cv2.VideoCapture(video_path)
num_frames = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
fps = v.get(cv2.CAP_PROP_FPS)
height = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))

print(f'video      :  {video_path}\n'
      f'num_frames :  {num_frames}\n'
      f'fps        :  {fps}\n'
      f'resolution :  {width} x {height}')

# %%
# 2. 提取帧
import datetime


def get_frame_index(time_str: str, fps: float):
    t = time_str.split(':')
    t = list(map(float, t))
    if len(t) == 3:
        td = datetime.timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    elif len(t) == 2:
        td = datetime.timedelta(minutes=t[0], seconds=t[1])
    else:
        raise ValueError(
            'Time data "{}" does not match format "%H:%M:%S"'.format(time_str))
    index = int(td.total_seconds() * fps)
    return index


# 起始时间、结束时间
time_start = '0:00'
time_end = '0:10'
ocr_start = get_frame_index(time_start, fps) if time_start else 0
ocr_end = get_frame_index(time_end, fps) if time_end else num_frames
num_ocr_frames = ocr_end - ocr_start
print(f'ocr_start       :  {ocr_start}\n'
      f'ocr_end         :  {ocr_end}\n'
      f'num_ocr_frames  :  {num_ocr_frames}')

# %%
# 3. 只保留画面中的字幕区域

# *** 调整字幕区域的高度，按比例 ***
h1, h2 = 0.86, 0.94
h1, h2 = int(height * h1), int(height * h2)

v.set(cv2.CAP_PROP_POS_FRAMES, ocr_start)
frames = [v.read()[1] for _ in range(num_ocr_frames)]
z_frames = [frame[h1:h2, :] for frame in frames]

# 预览一下
title = 'preview'
cv2.startWindowThread()
cv2.namedWindow(title)
for idx, img in enumerate(z_frames):
    tmp_img = img.copy()
    cv2.putText(tmp_img, f'idx:{idx}', (5, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(title, tmp_img)
    cv2.imshow(title, img)
    cv2.waitKey(50)
cv2.destroyWindow(title)
cv2.destroyAllWindows()

# %%
# 4. 去除相似度较高的帧，保留关键帧

# 设置阈值
mse_threshold = 100

from skimage.metrics import mean_squared_error

k_frames = [{'start': 0,
             'end': 0,
             'frame': z_frames[0],
             'text': ''}]

for idx in range(1, num_ocr_frames):
    img1 = z_frames[idx - 1]
    img2 = z_frames[idx]

    mse = mean_squared_error(img1, img2)
    # print(idx, mse)

    if mse < mse_threshold:
        k_frames[-1]['end'] = idx
    else:
        k_frames.append({'start': idx,
                         'end': idx,
                         'frame': z_frames[idx],
                         'text': ''})

for kf in k_frames:
    print(f"{kf['start']} --> {kf['end']} : {kf['text']}")

# %%
# 5. 识别字幕

import pytesseract

config = f'--tessdata-dir "{tessdata_dir}" --psm 7'

for idx, kf in enumerate(k_frames):
    # 识别为字符串
    ocr_str = pytesseract.image_to_string(kf['frame'], lang=lang, config=config)
    ocr_str = ocr_str.strip().replace(' ', '')

    if ocr_str:
        k_frames[idx]['text'] = ocr_str
        print(f"{kf['start']} --> {kf['end']} : {kf['text']}")

print([k_frames.remove(kf) for kf in k_frames if not kf['text']])

# %%
# 6. 格式化字幕

for kf in k_frames:
    print(f"{kf['start']} --> {kf['end']} : {kf['text']}")


def get_srt_timestamp(frame_index: int, fps: float):
    td = datetime.timedelta(seconds=frame_index / fps)
    ms = td.microseconds // 1000
    m, s = divmod(td.seconds, 60)
    h, m = divmod(m, 60)
    return '{:02d}:{:02d}:{:02d},{:03d}'.format(h, m, s, ms)


for kf in k_frames:
    time1 = get_srt_timestamp(kf['start'], fps)
    time2 = get_srt_timestamp(kf['end'], fps)

    print(f"{time1} --> {time2}\n{kf['text']}\n")
