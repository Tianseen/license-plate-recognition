import argparse

import torch.backends.cudnn as cudnn

from models.experimental import *
from utils.datasets import *
from utils.utils import *
from models.LPRNet import *


# vdoDet(), imgDet() and licCut() 


def vdoDet(source,save_img=False):
    opt = parse_opt()
    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz = \
        opt.classify, opt.vid_output, opt.vid_source, opt.det_weights, opt.rec_weights,  opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov3 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier 也就是rec 字符识别
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))
        print("load rec pretrained model successful!")
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred, plat_num = apply_classifier(pred, modelc, img, im0s)

        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    *xyxy, conf, cls = de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        lb = ""
                        for a, i in enumerate(lic_plat):
                            lb += CHARS[int(i)]
                        label = '%s %.2f' % (lb, conf)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            '''
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            '''
            
            # Save results (image with detections) 
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # rec_result video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(im0)

    if save_txt or save_img:
        if os.listdir(out):
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'Windows':  # windows系统
                os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    return im0

def imgDet(source, save_img=False):
    opt = parse_opt()
    img_list = []
    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz = \
        opt.classify, opt.img_output, opt.img_source, opt.det_weights, opt.rec_weights,  opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov3 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier 也就是rec 字符识别
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))
        print("load rec pretrained model successful!")
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size demo
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run demo
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred, plat_num = apply_classifier(pred, modelc, img, im0s)

        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for de, lic_plat in zip(det, plat_num):
                    *xyxy, conf, cls = de

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        lb = ""
                        for a, i in enumerate(lic_plat):
                            lb += CHARS[int(i)]
                        label = '%s %.2f' % (lb, conf)
                        im0 = plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            '''
            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
            '''
            # Save results (image with detections)
            if save_img:

                cv2.imwrite(save_path, im0)  #如果不需要保存到本地就注释掉这一行
                img_list.append(im0)
                '''
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # rec_result video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)'''

    if save_txt or save_img:
        #如果rec_result文件夹下有文件,就打印下面的话
        if os.listdir(out):
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'Windows':  # windows系统
                os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))

    '''
    #创建预训练模型
    create_pretrained(opt.weights, opt.weights)  
    '''
    return img_list


'''
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify', nargs='+', type=str, default=True, help='True rec')
    parser.add_argument('--det-weights', nargs='+', type=str, default=r'weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--rec-weights', nargs='+', type=str, default=r'weights/LPRNet__iteration_500.pth', help='model.pt path(s)')
    parser.add_argument('--vid_source', type=str, default=r'src/videos', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--vid_output', type=str, default=r'src/rec_result/videos', help='rec_result folder')  # rec_result folder
    parser.add_argument('--img_source', type=str, default=r'src/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img_output', type=str, default=r'src/rec_result/photos', help='rec_result folder')  # rec_result folder
    parser.add_argument('--img-size', type=int, default=640, help='demo size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented demo')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)
    return opt
'''

def parse_opt(request):
    classify = request.GET.get('classify', 'True')  # 默认值可以根据实际情况调整
    det_weights = request.GET.getlist('det-weights', ['weights/best.pt'])
    rec_weights = request.GET.getlist('rec-weights', ['weights/LPRNet__iteration_500.pth'])
    vid_source = request.GET.get('vid_source', 'src/videos')
    vid_output = request.GET.get('vid_output', 'src/rec_result/videos')
    img_source = request.GET.get('img_source', 'src/images')
    img_output = request.GET.get('img_output', 'src/rec_result/photos')
    img_size = int(request.GET.get('img-size', '640'))
    conf_thres = float(request.GET.get('conf-thres', '0.4'))
    iou_thres = float(request.GET.get('iou-thres', '0.5'))
    device = request.GET.get('device', '')
    view_img = bool(request.GET.get('view-img', ''))
    save_txt = bool(request.GET.get('save-txt', ''))
    classes = [int(c) for c in request.GET.getlist('classes')] if 'classes' in request.GET else None
    agnostic_nms = bool(request.GET.get('agnostic-nms', ''))
    augment = bool(request.GET.get('augment', ''))
    update = bool(request.GET.get('update', ''))

    opt = {
        'classify': classify,
        'det_weights': det_weights,
        'rec_weights': rec_weights,
        'vid_source': vid_source,
        'vid_output': vid_output,
        'img_source': img_source,
        'img_output': img_output,
        'img_size': img_size,
        'conf_thres': conf_thres,
        'iou_thres': iou_thres,
        'device': device,
        'view_img': view_img,
        'save_txt': save_txt,
        'classes': classes,
        'agnostic_nms': agnostic_nms,
        'augment': augment,
        'update': update,
    }

    print(opt)
    return opt


'''
if __name__ == '__main__':
    opt = parse_opt()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov3.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt','yolov3-fixed.pt']:
                detect()
                create_pretrained(opt.weights, opt.weights)
        else:
            detect()
'''

def parse_opt(request):
    classify = request.GET.get('classify', 'True')  # 默认值可以根据实际情况调整
    det_weights = request.GET.getlist('det-weights', ['weights/best.pt'])
    rec_weights = request.GET.getlist('rec-weights', ['weights/LPRNet__iteration_500.pth'])
    img_path = request.GET.get('img_path', 'src/images/image.jpg')
    img_output = request.GET.get('img_output', 'src/rec_result/photos')
    img_size = int(request.GET.get('img-size', '640'))
    conf_thres = float(request.GET.get('conf-thres', '0.4'))
    iou_thres = float(request.GET.get('iou-thres', '0.5'))
    device = request.GET.get('device', '')
    view_img = bool(request.GET.get('view-img', ''))
    save_txt = bool(request.GET.get('save-txt', ''))
    classes = [int(c) for c in request.GET.getlist('classes')] if 'classes' in request.GET else None
    agnostic_nms = bool(request.GET.get('agnostic-nms', ''))
    augment = bool(request.GET.get('augment', ''))
    update = bool(request.GET.get('update', ''))

    opt = {
        'classify': classify,
        'det_weights': det_weights,
        'rec_weights': rec_weights,
        'img_path': img_path,
        'img_output': img_output,
        'img_size': img_size,
        'conf_thres': conf_thres,
        'iou_thres': iou_thres,
        'device': device,
        'view_img': view_img,
        'save_txt': save_txt,
        'classes': classes,
        'agnostic_nms': agnostic_nms,
        'augment': augment,
        'update': update,
    }

    print(opt)
    return opt

def licCut(img_path, save_img=False):
    opt = parse_opt()
    classify, out, source, det_weights, rec_weights, view_img, save_txt, imgsz, img_path = \
        opt.classify, opt.img_output, opt.img_source, opt.det_weights, opt.rec_weights,  opt.view_img, opt.save_txt, opt.img_size, opt.img_path
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    img_list = []
    label_list = []
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete rec_result folder
    os.makedirs(out)  # make new rec_result folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov3 model
    model = attempt_load(det_weights, map_location=device)  # load FP32 model
    print("load det pretrained model successful!")
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier 也就是rec 字符识别
    if classify:
        modelc = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0).to(device)
        modelc.load_state_dict(torch.load(rec_weights, map_location=torch.device('cpu')))
        print("load rec pretrained model successful!")
        modelc.to(device).eval()

    #load image data
    save_img = True
    dataset = LoadImages(img_path, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    #run
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap, s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference  find license location by loyov3
        t1 = torch_utils.time_synchronized()
        pred = model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred, plat_num = apply_classifier(pred, modelc, img, im0s)

        t2 = torch_utils.time_synchronized()

        # Process detections   chioce between one or more
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s
            
            #create dir for save
            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if det is not None and len(det):
                # Write results
                for de, lic_plat in zip(det, plat_num):
                    *xyxy, conf, cls = de
                    lb = ""
                    for a, i in enumerate(lic_plat):
                        lb += CHARS[int(i)]

                    label = '%s' % (lb)    #only label

                # Rescale boxes from img_size to im0 size
                for det in scale_coords(img.shape[2:], det[:, :4], im0.shape).round():
                    x_min, y_max, x_max, y_min = map(int, det)
                    im0 = Image.fromarray(im0)
                    im0 = im0.crop((x_min, y_max, x_max, y_min))
                    im0 = im0.resize((94, 24), Image.Resampling.LANCZOS)
                    im0 = np.asarray(im0)
                    
            # Print time (demo + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            
            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)     #如果不需要保存到本地就注释掉这一行

    if save_txt or save_img:
        #如果rec_result文件夹下有文件,就打印下面的话
        if os.listdir(out):
            print('Results saved to %s' % os.getcwd() + os.sep + out)
            if platform == 'Windows':  # windows系统
                os.system('open ' + save_path)
    print('Done. (%.3fs)' % (time.time() - t0))

    return im0, label