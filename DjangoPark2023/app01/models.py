from django.db import models

# Create your models here.

class vip_uer(models.Model):
    name = models.CharField(max_length=32, verbose_name='姓名')
    carnum = models.CharField(max_length=32, unique=True, verbose_name='车牌号')
    phone = models.CharField(max_length=32, verbose_name='手机号')
    begintime = models.DateTimeField(auto_now=False, auto_now_add=False)
    endtime = models.DateTimeField(auto_now=False, auto_now_add=False)

class car_record(models.Model):
    carnum = models.CharField(max_length=32, verbose_name='车牌号')
    intime = models.DateTimeField(auto_now=False, auto_now_add=False, verbose_name='入场时间')
    outtime = models.DateTimeField(auto_now=False, auto_now_add=False, verbose_name='出场时间', null=True, blank=True)
    paytime = models.DateTimeField(auto_now=False, auto_now_add=False ,verbose_name='收费时间', null=True, blank=True)
    amount = models.IntegerField(verbose_name='收费金额', null=True, blank=True)
class License_plate(models.Model):
    car_img = models.ImageField(upload_to='car_imgs',unique=True, blank=True, null=True)
    car_num = models.CharField(max_length=32, unique=True, verbose_name='车牌号', null=True)
    color = models.CharField(max_length=32, verbose_name='车牌颜色')