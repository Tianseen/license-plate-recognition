from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
import datetime
from app01 import models
from .models import vip_uer,car_record
from carDetTools import licCut, parse_opt
# Create your views here.

def Recharge(request):
    if request.method == 'POST':
        chargedays = int( request.POST.get('chargedays') )
        carnum = request.POST.get('carnum')
        print(chargedays)
        days = datetime.timedelta(days=chargedays)
        if models.vip_uer.objects.filter(carnum=carnum):
            vip = vip_uer.objects.get(carnum=carnum)
            endtime = vip.endtime + days
            print(vip.endtime)
            models.vip_uer.objects.update(carnum=carnum, endtime = endtime)
        else:
            begtime = datetime.datetime.now()
            endtime = begtime + days
            models.vip_uer.objects.create(carnum=carnum, begintime = begtime ,endtime=endtime)
        return HttpResponse("充值成功！")
    else:
        return render(request, 'recharge.html')

def Park_discern(request):
    try:
        opt = parse_opt(request)
        lic, label = Park_discern(opt)
        color = 0
        # 处理结果
        return label, color
    except Exception as e:
        # 处理异常
        print(f"车牌识别发生错误: {e}")
        return '未识别到车牌号', '未识别到颜色'


def car_in(request):
    if request.method == 'POST':
        # 读取图片
        img = request.FILES.get('car_img')
        if img == None:
            # 没有选择图片，而直接点击检测
            error = '请选择一张图片！'
            return render(request, 'car_in.html', {'error': error})
        else:
            try:
                # 将图片数据存起来,并返回图片地址
                new_car = models.License_plate.objects.create(car_img=img)
                id = new_car.id

                #从数据库读取图片地址
                new_car = models.License_plate.objects.get(id=id)
                url = './media/' + str(new_car.car_img)
                # 调用接口识别车牌
                res = Park_discern(url)
                #车牌号
                carnum = res[0]
                color = 0
                #车牌颜色
                #color = res['words_result']['color']
                try:
                    # 车牌是否识别过
                    is_carnum = models.License_plate.objects.get(car_num=carnum)
                    if is_carnum:
                        #识别过了的直接从数据库读取历史数据并删除当前存储的图片数据和文件
                        new_car.car_img = is_carnum.car_img
                        print(new_car.id )
                        models.License_plate.objects.filter(id=new_car.id ).delete()
                except models.License_plate.DoesNotExist:
                    # 没识别过，则保存车牌和颜色信息
                    new_car.color = color
                    new_car.car_num = carnum
                    new_car.save()
                    # return redirect('carnum_add')
                    print(new_car.car_img)
                return render(request,'car_in.html',{'carport_url':new_car.car_img,'carnum':carnum,'color':color})
                #return render(request,'car_in.html',{'carport_url':new_car.car_img,'carnum':carnum})
            except Exception as e:
                # 记录异常信息
                print(f'Exception: {e}')
                return HttpResponse(f'识别发生错误！错误信息: {e}')
    return render(request, 'car_in.html',{'carport_url':'car_imgs/intro.jpg'})

from django.utils import timezone

def carin_update(request):
    if request.method == 'POST':
        carnum = request.POST.get('carnum')
        new_intime = timezone.now()
        print(new_intime)
        models.car_record.objects.create(carnum=carnum, intime=new_intime)
        if models.vip_uer.objects.filter(carnum=carnum):
            vip = vip_uer.objects.get(carnum=carnum)
            endtime = vip.endtime
            remain_days = endtime - new_intime
            remain_days = str(remain_days.days)
            context = '会员车辆' + carnum + '欢迎您！'  + '剩余会员天数:' + remain_days
            print(context)
            return HttpResponse(context)
        else:
            return HttpResponse("临时车辆，欢迎入场！")
def car_out(request):
    if request.method == 'POST':
        # 读取图片
        img = request.FILES.get('car_img')
        if img == None:
            # 没有选择图片，而直接点击检测
            error = '请选择一张图片！'
            return render(request, 'car_in.html', {'error': error})
        else:
            try:
                # 将图片数据存起来
                new_car = models.License_plate.objects.create(car_img=img)
                # 定义读取图片函数
                def get_file_content(filePath):
                    with open(filePath, 'rb') as fp:
                        return fp.read()
                #生成图片地址
                url = './media/' + str(new_car.car_img)
                # 读取图片信息
                image = get_file_content(url)
                # 调用接口识别车牌
                res = Park_discern(image)
                #车牌号
                carnum = res['words_result']['number']
                #车牌颜色
                color = res['words_result']['color']
                try:
                    # 车牌是否识别过
                    is_carnum = models.License_plate.objects.get(car_num=carnum)
                    if is_carnum:
                        #识别过了的直接从数据库读取历史数据并删除当前存储的图片数据和文件
                        new_car.car_img = is_carnum.car_img
                        print(new_car.id )
                        models.License_plate.objects.filter(id=new_car.id ).delete()
                except models.License_plate.DoesNotExist:
                    # 没识别过，则保存车牌和颜色信息
                    new_car.color = color
                    new_car.car_num = carnum
                    new_car.save()
                    # return redirect('carnum_add')
                    print(new_car.car_img)
                return render(request,'car_out.html',{'carport_url':new_car.car_img,'carnum':carnum,'color':color})
            except Exception as e:
                return HttpResponse('识别发生错误！')
    return render(request, 'car_out.html',{'carport_url':'car_imgs/intro.jpg'})

def carout_update(request):
    if request.method == 'POST':
        carnum = request.POST.get('carnum')
        new_outtime = timezone.now()
        print(new_outtime)
        new_car_record = car_record.objects.filter(carnum=carnum).order_by('outtime')
        out_car = new_car_record[0]
        out_car.carnum = carnum
        out_car.outtime = new_outtime
        out_car.save()
        if models.vip_uer.objects.filter(carnum=carnum):
            vip = vip_uer.objects.get(carnum=carnum)
            print(vip)
            # now = datetime.now(timezone.utc)
            endtime = vip.endtime
            remain_days = endtime - new_outtime
            remain_days = str(remain_days.days)
            print(remain_days)
            context = '会员车辆' + carnum + '一路顺风！'  + '剩余会员天数:' + remain_days
            print(context)
            return HttpResponse(context)
        else:
            return HttpResponse("临时车辆，一路顺风！")
