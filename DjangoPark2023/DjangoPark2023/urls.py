"""
URL configuration for DjangoPark2023 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# DjangoPark2023/urls.py
from django.contrib import admin
from django.urls import path
from app01 import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('recharge/', views.Recharge, name='recharge'),  # 添加充值页面的URL映射
    path('car_in/', views.car_in, name='car_in'),  # 添加车辆入场页面的URL映射
    path('carin_update/', views.carin_update, name='carin_update'),  # 添加车辆入场更新页面的URL映射
    path('car_out/', views.car_out, name='car_out'),  # 添加车辆出场页面的URL映射
    path('carout_update/', views.carout_update, name='carout_update'),  # 添加车辆出场更新页面的URL映射
]

