from django.contrib import admin
# Register your models here.
# 别忘了导入models
from app01.models import vip_uer
# 注册模型到admin中
admin.site.register(vip_uer)
