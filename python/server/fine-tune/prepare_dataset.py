import os
import cv2
import xml.etree.ElementTree as ET

from pathlib import Path


# Print iterations progress
def printProgressBar(
    iteration,
    total,
    prefix='',
    suffix='',
    decimals=1,
    length=100,
    fill='█',
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:."+str(decimals)+"f}").format(100*(iteration/float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


# Parse XML file
def parse_xml(xml_paths):
    boxes = {}

    offset_frame0 = 0
    offset_id0 = 0
    offset_frame = 0
    offset_id = 0
    for xml_path in xml_paths:
        print(xml_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        offset_frame += offset_frame0
        offset_id += offset_id0
        print('offset_frame0 = ', offset_frame0, 'offset_frame = ', offset_frame)
        w = int(root.find('meta').find('original_size').find('width').text)
        h = int(root.find('meta').find('original_size').find('height').text)

        for track in root.findall('track'):
            id = int(track.get('id'))
            offset_id0 = id
            cls = track.get('label')
            if cls != 'car':
                continue
            cnt = 0
            for box in track.findall('box'):
                if cnt >= len(track.findall('box')) - 30 * 4:
                    break
                cnt += 1
                if box.get('outside') == "0" and cnt % 4 == 0:
                    offset_frame0 = int(box.get('frame'))
                    iframe = (int(box.get('frame')) + 1) #// 4 + 1
                    x1 = float(box.get('xtl'))/w
                    y1 = float(box.get('ytl'))/h
                    x2 = float(box.get('xbr'))/w
                    y2 = float(box.get('ybr'))/h

                    speed = int(box.find('attribute').text)

                    boxes[iframe + offset_frame] = (
                        id + offset_id, (x1, y1, x2-x1, y2-y1), speed, cls
                    )
        offset_frame += 1000000
    return boxes


videos = [
    'FILE0001.mp4',
    'FILE0002.mp4',
    # 'FILE0004.mp4',
    # 'FILE0005.mp4',
    # 'FILE0008.mp4',
    # 'FILE0009.mp4',
    'FILE0010.mp4',
    # 'FILE0019.mp4',
    # 'FILE0026.mp4',
    # 'FILE0027.mp4',
    'FILE0030.mp4',
    # 'FILE0031.mp4',
]
videos_path = '../../../videos/'

dst_dir = Path("yolo-dataset")
train_dir = dst_dir / "train"
train_images_dir = train_dir / "images"
train_labels_dir = train_dir / "labels"
val_dir = dst_dir / "val"
val_images_dir = val_dir / "images"
val_labels_dir = val_dir / "labels"

dst_dir.mkdir(exist_ok=True, parents=True)
train_dir.mkdir(exist_ok=True, parents=True)
train_images_dir.mkdir(exist_ok=True, parents=True)
train_labels_dir.mkdir(exist_ok=True, parents=True)
val_dir.mkdir(exist_ok=True, parents=True)
val_images_dir.mkdir(exist_ok=True, parents=True)
val_labels_dir.mkdir(exist_ok=True, parents=True)

# videos_path = '/media/javad/24D69A46D69A17DE/code/vehicle_speed_project_2/videos/'
video_paths = [videos_path + v for v in videos]

json_paths = [vp.replace('mp4', 'xml') for vp in video_paths]

annotations = parse_xml(json_paths)

total = max(annotations.keys())

fo0 = 0
fo = 0
for video_path in video_paths:
    # video_path_mp4 = video_path.replace('ASF', 'mp4')
    # # Convert the video to MP4 format
    # try:
    #     video_clip = VideoFileClip(video_path)
    #     video_clip.write_videofile(
    #         video_path_mp4, codec="libx264", audio_codec="aac")
    #     print(f"Conversion successful: {video_path_mp4} has been created.")
    # except Exception as e:
    #     print(f"An error occurred: {e}")
    #     break
    # finally:
    #     video_clip.close()

    cap = cv2.VideoCapture(video_path)

    vid_name = video_path.split("/")[-1]
    print(vid_name)

    prev_gray = None
    prev_pts = None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video_len= {video_len} \n")
    fn = 0
    fo += fo0
    print('fo0 = ', fo0, 'fo = ', fo, '\n')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        iframe = fn + fo
        if iframe in annotations:
            fo0 = fn
            id, (x1, y1, w, h), speed, cls = annotations[iframe]

            if id % 10 == 0:
                dir = val_dir
            else:
                dir = train_dir

            cv2.imwrite(str(dir / f"images/{iframe:04d}.jpg"), frame)
            f = open(str(dir / f"labels/{iframe:04d}.txt"), "w")
            f.write(f"2 {x1} {y1} {w} {h}\n")
            f.close()

            x2 = x1 + w
            y2 = y1 + h
            cv2.rectangle(frame,
                        (int(x1*W), int(y1*H)),
                        (int(x2*W), int(y2*H)), (0, 255, 0), 2)
            
            # Display the processed frame
            cv2.imshow('Processed Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.waitKey(0)

        if iframe % 300 == 0:
            printProgressBar(
                fn,
                video_len,
                prefix=f'Processing {vid_name}:',
                suffix='Complete',
                length=30,
            )
        fn += 1

    fo += 1000000
    cap.release()

    # # Check if the file exists before removing it
    # if os.path.exists(video_path_mp4) and 'mp4' in video_path_mp4:
    #     os.remove(video_path_mp4)
    #     print(f"{video_path_mp4} has been removed successfully.")
    # else:
    #     print(f"{video_path_mp4} does not exist.")

